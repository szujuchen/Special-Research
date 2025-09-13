import argparse
import datetime
import json
import random
from sched import scheduler
import time
from pathlib import Path
import sys

import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import pytorch_warmup as warmup
from dataset.dataset_img import IMG_DATA
from model.tokenizer import vitrqfc
from model.DictTree import TreeNode
from utils.utils import train_transform, get_trainable_params, residualquantizer, InfoNCELoss
from utils.logger import get_logger
import os
import pickle
from model.CLIP.clip import clip
import scheduler
import tqdm


@torch.no_grad()
def get_features(Dataset, model):
    loader = DataLoader(Dataset, batch_size=32)
    with torch.no_grad():
        for i, (images,_,_) in enumerate(loader):
            features = model(images.cuda())[:,0,:]
            if i==0:
                feat = features.cpu()
            else:
                feat = torch.cat((feat, features.cpu()), 0)
    del loader
    return feat.numpy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Tokenizer')
    parser.add_argument('--data_dir', default='data', type=str, help='datasets path')
    parser.add_argument('--feats', default='', type=str)
    parser.add_argument('--output_dir', default='', type=str)
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=128, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=200, type=int, help='train epoch number')
    parser.add_argument('--num_workers', default=8, type=int, help='dataloader worker')
    parser.add_argument('--id_len', default=4, type=int, help='image token length')
    parser.add_argument('--codebook_size', default=8, type=int, help='codebook size')
    parser.add_argument('--contrastive', default=0, type=float, help='percentage of contrastive loss range[0-1]')
    opt = parser.parse_args()

    data_dir = opt.data_dir
    lr = opt.lr
    batch_size = opt.batch_size
    num_epochs = opt.num_epochs
    if opt.output_dit == '':
        output_dir = f"{data_dir}_model"
    else:
        output_dir = opt.output_dir
    num_workers = opt.num_workers
    alpha = opt.contrastive
    if alpha < 0 or alpha > 1:
        print("Argument INVALID contrastive should be in range of 0 - 1")
        sys.exit()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_classes = len([f for f in os.listdir(os.path.join(data_dir, 'images/train')) if os.path.isdir(os.path.join(data_dir, 'images/train', f))])
    Dataset = IMG_DATA(data_dir, 'db',transform=train_transform(256,224))

    model = vitrqfc(dec_depth=12, num_classes=num_classes)
    mm, preprocess = clip.load('ViT-B/16')
    mm=mm.type(torch.float32)
    model.encoder = mm.visual
    model = model.cuda()

    if opt.feats != '':
        feats = np.load(opt.feats)
    else:
        feats = get_features(Dataset, model.encoder)


    train_loader = DataLoader(Dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    optimizer = AdamW([{'params': get_trainable_params(model.fc)},
                    {'params': get_trainable_params(model.fc_rq), 'lr':1*lr}, 
                    {'params':get_trainable_params(model.encoder), 'lr': 0.01*lr}
                    ], lr=lr, betas=(0.9, 0.96), eps=1e-08, weight_decay=0.05)
    scheduler = scheduler.CosineLRScheduler(optimizer=optimizer,
                                                t_initial=50,
                                                lr_min=1e-6,
                                                warmup_t=20,
                                                warmup_lr_init=5e-7,
                                                cycle_decay=0.5
                                                )
    criterion = nn.CrossEntropyLoss(ignore_index=-1) 
    cos = nn.CosineSimilarity()
    contrastive = InfoNCELoss()

    logger = get_logger('tokenizer.log')
    for i in range(num_epochs):
        print('start epoch',i)
        x_q, rq_code = residualquantizer(feats,opt.id_len,opt.codebook_size)

        for j,(img,target,idx) in enumerate(train_loader):
            img = img.cuda()
            target = target.unsqueeze(-1).cuda()
            z_q = [torch.tensor(np.array([x_q[l][k] for k in idx])).float().cuda() for l in range(len(x_q))]
            z, loss_quant, output, output_rq = model(img, z_q)
            
            output_rq = [output_rq[k].reshape(-1,num_classes) for k in range(len(output_rq))]
            output = output.reshape(-1,num_classes)
            target = target.reshape(-1)
            target = target - 1

            #IRGEN loss
            loss_ce_rq = []
            for k in range(len(output_rq)):
                loss_ce_rq.append(criterion(output_rq[k], target))
            loss_ce = criterion(output, target)
            ce_loss = 1e-7*(sum(loss_ce_rq)+ sum(loss_quant)) + loss_ce 
            #contrastive loss
            reconstructed = z_q[-1]
            contrastive_loss = contrastive(z, reconstructed)
    
            loss = (1-alpha)*ce_loss + (alpha)*contrastive_loss
            
            running_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if j%10 == 9:
                logger.info('Epoch:[{}/{}]\t step={}\t loss={:.4f}\t loss_ce={:.2f}\t loss_ce_rq_0={:.2f}\tloss_ce_rq_1={:.2f}\tloss_ce_rq_2={:.2f}\tloss_ce_rq_3={:.2f}\t loss_quant={:.2f}\t lr={:.8f}'.format(i , num_epochs, j+1, loss, loss_ce, loss_ce_rq[0], loss_ce_rq[1], loss_ce_rq[2], loss_ce_rq[3], sum(loss_quant), optimizer.param_groups[0]['lr'] ))
            percent=(j+1)/len(train_loader)
            scheduler.step(i+percent)
        
        scheduler.step(i+1)
        feats = get_features(Dataset, model.encoder)
        
        print('epoch:', i, 'loss:', loss)
        if i%50 == 49:
            state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':i+1, 'loss':running_loss}
            torch.save(state, os.path.join(output_dir, 'tok_e{}.pkl').format(i+1))
            np.save(os.path.join(output_dir, 'tok_feats_e{}.npy').format(i+1),feats)
    
    state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':i+1, 'loss':running_loss}
    torch.save(state, os.path.join(output_dir, 'tok.pkl').format(i+1))
    np.save(os.path.join(output_dir, 'tok_feats.npy').format(i+1),feats)