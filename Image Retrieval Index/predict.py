import argparse
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from model.IRGen import IRGen
from dataset.config import config_gnd
from dataset.dataset_img import TEST_IMG
from utils.logger import get_logger
from utils.utils import test_transform
from model.CLIP.clip import clip
import pickle
from PIL import Image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test IRGen')
    parser.add_argument('--data_dir', default='data', type=str, help='datasets path')
    parser.add_argument('--codes', default='codes.pkl', type=str, help='image identifier')
    parser.add_argument('--model_dir', default='', type=str)
    parser.add_argument('--query_img', required=True, type=str)
    parser.add_argument('--beam_size', default=5, type=int, help='test beam size')
    parser.add_argument('--output_dir', default='', type=str)
    opt = parser.parse_args()

    query = opt.query_img
    data_dir = opt.data_dir
    codes = opt.codes
    beam_size = opt.beam_size
    if opt.model_dir == "":
        model_dir = f"{data_dir}_model"
    else:
        model_dir = opt.model_dir
    if opt.output_dir == "":
        output_dir = f"{data_dir}_pred"
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

        Dataset = TEST_IMG(query, transform=test_transform(256,224))
        test_loader = DataLoader(Dataset, batch_size=1)

        # should just have one test image
        for i, img in enumerate(test_loader):
            img = img.cuda()
            predictions = model.beam_search(img,k=beam_size,k_tree=k_tree,ids=ids)[0]
    
    cfg = config_gnd(data_dir, 'retrieval.pkl')
    gnd = cfg['gnd']
    for i, pred in enumerate(predictions):
        fn = os.path.join('data/images', cfg['imlist'][pred])
        result = Image.open(fn)

        save_fn = os.path.join(output_dir, f"pred{i}.jpg")
        result.save(save_fn)


