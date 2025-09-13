import argparse
import datetime
import json
import random
from sched import scheduler
import time
from pathlib import Path
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import pytorch_warmup as warmup
from dataset.dataset_img import IMG_DATA
from model.CLIP.clip import clip
from model.IRGen import IRGen
from model.DictTree import TreeNode
import utils.utils as utils
from utils.utils import LabelSmoothingCrossEntropy, train_transform, get_trainable_params, ContrastiveTokenLoss
from utils.logger import get_logger
import os
import pickle
import scheduler

if __name__ == '__main__':
    utils.init_distributed_mode('env://')
    # os.environ["MASTER_PORT"] = "29400"
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["WORLD_SIZE"] = "1"
    # os.environ["RANK"] = "0"
    # torch.distributed.init_process_group(backend="nccl", init_method='env://')
    # # torch.distributed.init_process_group(backend="nccl")
    local_rank = utils.get_rank()
    # torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = utils.get_world_size()

    parser = argparse.ArgumentParser(description='Train IRGen')
    parser.add_argument('--data_dir', default='data', type=str, help='datasets path')
    parser.add_argument('--cls_codes_file', default='cls_codes.pkl', type=str)
    parser.add_argument('--codes', default='codes.pkl', type=str, help='image identifier')
    parser.add_argument('--pretrained_model', default=None, type=str)
    parser.add_argument('--output_dir', default='', type=str)
    parser.add_argument('--lr', default=4e-5, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=64, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=200, type=int, help='train epoch number')
    parser.add_argument('--num_workers', default=8, type=int, help='train data loader workers')
    parser.add_argument('--smoothing', default=0.1, type=float, help='label smoothing')
    parser.add_argument('--contrastive', default=1, type=float)
    opt = parser.parse_args()

    if opt.contrastive < 0 or opt.contrastive > 1:
        print("Invalid argument. contrastive should be in range 0-1")
        sys.exit()

    # args parse
    data_dir = opt.data_dir
    codes_file = opt.codes
    pretrained_model = opt.pretrained_model
    if opt.output_dir == '':
        output_dir = f"{data_dir}_model"
    else:
        output_dir = opt.output_dir
    cls_gnd_file_name = opt.cls_codes_file
    num_epochs, batch_size, lr = opt.num_epochs, opt.batch_size, opt.lr
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(data_dir, codes_file),'rb')as f:
        codes = pickle.load(f)
    k_tree = codes['dict_tree']
    mapping = codes['mapping']
    codebook = codes['codebook']
    ids = torch.tensor(mapping).cuda()
    id_length = mapping.shape[-1]
    num_classes = np.unique(mapping).shape[0]
    Dataset = IMG_DATA(data_dir, 'db', transform=train_transform(256,224))

    train_sampler = DistributedSampler(Dataset, shuffle=True)
    model = IRGen(dec_depth=12, num_classes=num_classes, id_len=id_length)
    mm, preprocess = clip.load('ViT-B/16')
    mm = mm.to('cpu')
    mm=mm.type(torch.float32)
    model.encoder = mm.visual
    if pretrained_model:
        checkpoint = torch.load(pretrained_model)
        state_dict = checkpoint['net']
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank], 
                                    output_device=local_rank)

    train_loader = DataLoader(Dataset, batch_size=batch_size, sampler=train_sampler, num_workers=opt.num_workers)

    optimizer = AdamW([{'params': get_trainable_params(model.module.decoder), 'lr':1*lr},
                    {'params':get_trainable_params(model.module.encoder), 'lr': 0.01*lr}
                    ], lr=lr, betas=(0.9, 0.96), eps=1e-08, weight_decay=0.05)
    scheduler = scheduler.CosineLRScheduler(optimizer=optimizer,
                                                t_initial=50,
                                                lr_min=1e-6,
                                                warmup_t=20,
                                                warmup_lr_init=5e-7,
                                                cycle_decay=0.5,
                                                # cycle_limit=8,
                                                )

    criterion = LabelSmoothingCrossEntropy()
    contrastive = ContrastiveTokenLoss(pad_id=-1)
    logger = get_logger('ar.log')
    with open(os.path.join(data_dir, 'retrieval.pkl'), 'rb') as fin:
        gnd = pickle.load(fin)
    class_idx = gnd['class_idx']
    cls = gnd['classes']

    with open(os.path.join(data_dir, cls_gnd_file_name), 'rb') as f:
        cls_gnd = pickle.load(f)

    Loss = []
    LR = []
    for i in range(num_epochs):
        train_loader.sampler.set_epoch(num_epochs)
        if dist.get_rank() == 0:
            print('start epoch',i)
        for j,(img,clss,id_train) in enumerate(train_loader):
            idx = [np.random.rand(1)*len(class_idx[int(clss[k])]) for k in range(len(img))]
            tgt = np.array([mapping[class_idx[int(clss[k])][int(idx[k])]] for k in range(len(img))]).astype(np.int32)
            tgt[np.where(tgt==-1)]=num_classes
            target = torch.tensor(tgt,dtype=torch.int64)

            img = img.cuda()
            target = target.cuda()
            output = model(img, tgt=target)
            output = output[:,:-1,:]
            # target shape [B, M]
            # output shape [B, M, V]
            ce_loss = criterion(output.reshape(-1,num_classes), target.reshape(-1), opt.smoothing)
            ct_loss = contrastive(output, target, cls_gnd, clss)
            

            loss = (1-opt.contrastive)*ce_loss + opt.contrastive*ct_loss
            running_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if j%10 == 9:
                if dist.get_rank() == 0:
                    logger.info('Epoch:[{}/{}]\t step={}\t loss={:.10f}\t lr={:.8f}'.format(i , num_epochs, j+1, loss, optimizer.param_groups[0]['lr'] ))
            percent=(j+1)/len(train_loader)
            scheduler.step(i+percent)
        scheduler.step(i+1)

        if dist.get_rank() == 0:
            print('epoch:', i, 'loss:', loss)
            if i>90 and i%50 == 49:
                state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':i+1, 'loss':running_loss}
                torch.save(state, os.path.join(output_dir,'ar_e{}.pkl').format(i+1))

        dist.barrier()

    state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':i+1, 'loss':running_loss}
    torch.save(state, os.path.join(output_dir,'ar.pkl').format(i+1))
