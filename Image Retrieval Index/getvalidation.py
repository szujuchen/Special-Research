import argparse
import os
from dataset.config import config_gnd
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description='Generate validation IRGen')
parser.add_argument('--pred_dir', default='', type=str, help='prediction path')
parser.add_argument('--data_dir', default='data', type=str, help='dataset path')
parser.add_argument('--size', default=5, type=int, help='visualize size')
opt = parser.parse_args()

dir, data_dir = opt.pred_dir, opt.data_dir
if dir == '':
    dir = f"{data_dir}_valid"

with open(os.path.join(dir, "predictions.npy"), "rb") as f:
    preds = np.load(f)

cfg = config_gnd(data_dir, 'retrieval.pkl')
val_size = min(opt.size, len(preds))

for ind in range(len(preds[0])):
    pred_dir = os.path.join(dir, str(ind))
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    src_fn = os.path.join("data/images", cfg['qimlist'][ind])
    src = Image.open(src_fn)
    fn = os.path.join(pred_dir, "test.jpg")
    src.save(fn)

    for i in range(val_size):
        id = preds[i][ind]
        res_fn = os.path.join("data/images", cfg['imlist'][id])
        res = Image.open(res_fn)

        if id in cfg['gnd'][ind]:
            fn = os.path.join(pred_dir, f"pred{i}_true.jpg")
        else:
            fn = os.path.join(pred_dir, f"pred{i}_false.jpg")
        res.save(fn)
