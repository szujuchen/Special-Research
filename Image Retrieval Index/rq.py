import argparse
import faiss
import numpy as np
import os
import pickle
from model.DictTree import TreeNode


parser = argparse.ArgumentParser(description='rq')
parser.add_argument('--features', default='', type=str)
parser.add_argument('--output_file', default='codes.pkl', type=str)
parser.add_argument('--data_dir', default='data', type=str)
parser.add_argument('--classes_output_file', default='cls_codes.pkl', type=str)
parser.add_argument('--id_len', default=4, type=int)
parser.add_argument('--codebook_size', default=8, type=int)
opt = parser.parse_args()

data_dir = opt.data_dir
if opt.features == "":
    features = f"{data_dir}_model/tok_feats.npy"
features = opt.features
outfile = opt.output_file
cls_outfile = opt.classes_output_file

data = np.load(features)

dim = data.shape[1]
m = opt.id_len
k = opt.codebook_size
pq = faiss.ProductQuantizer(dim, 1, k)
x_q=[]
for i in range(m):
    print(i)
    pq.train(data)
    codes = pq.compute_codes(data)
    if i == 0:
        rq_codes = codes
        codebook = faiss.vector_to_array(pq.centroids).reshape(pq.M, pq.ksub, pq.dsub)
        datarec = pq.decode(codes)
    else:
        rq_codes = np.concatenate((rq_codes,codes),axis=1)
        codebook = np.concatenate((codebook,faiss.vector_to_array(pq.centroids).reshape(pq.M, pq.ksub, pq.dsub)),axis=0)
        datarec += pq.decode(codes)
    x_q.append(datarec.copy())
    data -= pq.decode(codes)

print(rq_codes.shape)
print(rq_codes[0])
print(codebook.shape)
print(type(codebook[0][1]))

kmeans_tree = TreeNode()
kmeans_tree.insert_many(rq_codes)

gnd = {'mapping':rq_codes, 'codebook':codebook, 'dict_tree': kmeans_tree}
with open(os.path.join(data_dir, outfile),'wb')as f:
    pickle.dump(gnd,f)

with open(os.path.join(data_dir, 'retrieval.pkl'), 'rb') as fin:
    data = pickle.load(fin)
    cls = data['classes']

classes = {}
for id, codes in enumerate(rq_codes):
    cl = cls[id]
    if cl not in classes.keys():
        classes[cl] = []
    classes[cl].append(codes)

classes_tree = {}
for cl, codes in classes.items():
    classes_tree[cl] = TreeNode()
    classes_tree[cl].insert_many(codes)

with open(os.path.join(data_dir, cls_outfile),'wb')as f:
    pickle.dump(classes_tree,f)


