import argparse
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from model.IRGen import IRGen
from model.DictTree import TreeNode
from dataset.dataset_img import IMG_DATA
from dataset.config import config_gnd
from utils.evaluate import compute_map
from utils.logger import get_logger
from utils.utils import test_transform
from model.CLIP.clip import clip
import pickle

@torch.no_grad()
def test(gnd, ks, ranks):
    map, aps, mpr, prs = compute_map(ranks, gnd, ks)

    total_rele = []
    for i in range(len(ranks[0])):
        flag = 0
        for j in range(ks[-1]):
            if ranks[j][i] in gnd[i]:
                flag += 1
        total_rele.append(flag)

    for k in ks:
        cnt = 0
        prec = 0
        recall = 0
        for i in range(len(ranks[0])):
            flag = 0
            for j in range(k):
                if ranks[j][i] in gnd[i]:
                    flag += 1
            prec += flag / k
            if total_rele[i] != 0:
                recall += flag / total_rele[i]
            if flag != 0:
                cnt += 1
        print('rank@{}:'.format(k), cnt/len(ranks[0]))
        print('precision@{}:'.format(k), prec/len(ranks[0]))
        print('recall@{}:'.format(k), recall/len(ranks[0]))

    return map, aps, mpr, prs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate IRGen')
    parser.add_argument('--data_dir', default='data', type=str, help='datasets path')
    parser.add_argument('--codes', default='codes.pkl', type=str, help='image identifier')
    parser.add_argument('--model_dir', default='', type=str)
    parser.add_argument('--beam_size', default=30, type=int, help='test beam size')
    parser.add_argument('--ks', default=[1,10,20,30], nargs='+', type=int)
    parser.add_argument('--output_dir', default='', type=str)
    opt = parser.parse_args()

    data_dir = opt.data_dir
    codes = opt.codes
    beam_size, ks = opt.beam_size, opt.ks
    if opt.model_dir == "":
        model_dir = f"{data_dir}_result"
    else:
        model_dir = opt.model_dir
    if opt.output_dir == "":
        output_dir = f"{data_dir}_valid"
    else:
        output_dir = opt.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with torch.no_grad():
        with open(os.path.join(data_dir, codes),'rb')as f:
            gnd = pickle.load(f)
        k_tree = gnd['dict_tree']
        mapping = gnd['mapping']
        ids = torch.tensor(mapping).cuda()
        
        id_length = ids.shape[-1]
        num_classes = np.unique(mapping).shape[0]
        model = IRGen(dec_depth=12, num_classes=num_classes, id_len=id_length)
 
        mm, preprocess = clip.load('ViT-B/16')
        mm = mm.to('cpu')
        mm=mm.type(torch.float32)
        model.encoder = mm.visual
        checkpoint = torch.load(model_dir)
        state_dict = checkpoint['net']
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model = model.cuda()

        Dataset = IMG_DATA(data_dir, 'query', transform=test_transform(256,224))

        test_loader = DataLoader(Dataset, batch_size=1)
        cfg = config_gnd(data_dir, 'retrieval.pkl')
        gnd = cfg['gnd']
        logger = get_logger('valid.log')

        for i, img in enumerate(test_loader):
            img = img.cuda()

            out = model.beam_search(img,k=beam_size,k_tree=k_tree,ids=ids)
            if i==0:
                ranks = np.array(out)
            else:
                ranks = np.concatenate((ranks,out),axis=0)
    
            logger.info('number:{} image \t preds{}'.format(i, out))
        ranks = np.asarray(ranks).T
        map, aps, mpr, prs = test(gnd, ks, ranks)
        logger.info('map:{}, mpr:{}'.format(map,mpr))
        
        with open(os.path.join(output_dir, 'predictions.npy'), "wb") as f:
            np.save(f, ranks)


