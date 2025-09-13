# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Misc functions.

Mostly copy-paste from torchvision references or other public repos like DETR:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""
import os
import sys
import time
import math
import random
import datetime
import subprocess
import warnings
from collections import defaultdict, deque

import numpy as np
import faiss
import torch
from torch import nn
import torch.distributed as dist
from PIL import ImageFilter, ImageOps
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomApply, ColorJitter, ToTensor, Normalize, RandomHorizontalFlip, RandomPerspective



def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(dist_url):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        rank, gpu, world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29400'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=world_size,
        rank=rank,
    )

    torch.cuda.set_device(gpu)
    print('| distributed init (rank {}): {}'.format(rank, dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(rank == 0)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (torch.Tensor, float, float, float, float) -> torch.Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False



def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.
    Arguments
    ---------
    ranks : zerro-based ranks of positive images
    nres  : number of positive images
    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap


def compute_map(ranks, gnd, kappas=[]):
    """
    Computes the mAP for a given set of returned results.
         Usage:
           map = compute_map (ranks, gnd)
                 computes mean average precsion (map) only
           map, aps, pr, prs = compute_map (ranks, gnd, kappas)
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query
         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    map = 0.
    nq = len(gnd) # number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)

        # sorted positions of positive and junk images (0 based)
        pos  = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgndj)]

        k = 0
        ij = 0
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # compute precision @ k
        pos += 1 # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j]); 
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)

    return map, aps, pr, prs


def multi_scale(samples, model):
    v = None
    for s in [1, 1/2**(1/2), 1/2]:  # we use 3 different scales
        if s == 1:
            inp = samples.clone()
        else:
            inp = nn.functional.interpolate(samples, scale_factor=s, mode='bilinear', align_corners=False)
        feats = model(inp).clone()
        if v is None:
            v = feats
        else:
            v += feats
    v /= 3
    v /= v.norm()
    return v


from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
def _convert_image_to_rgb(image):
    return image.convert("RGB")    

def train_transform(n1,n2):
    return Compose([
        Resize(n1, interpolation=3),
        RandomCrop(n2),
        # RandomPerspective(fill=0, p=1, distortion_scale=0.5),
        _convert_image_to_rgb,
        ToTensor(),
    #    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
       Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
def test_transform(n1,n2):
    return Compose([
        Resize(n1, interpolation=3),
        CenterCrop(n2),
        # RandomPerspective(fill=0, p=1, distortion_scale=0.5),
        _convert_image_to_rgb,
        ToTensor(),
    #    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
       Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def get_trainable_params(model):
    return [params for params in model.parameters() if params.requires_grad]

def residualquantizer(data, id_len, k):
    dim = data.shape[1]
    m = id_len
    pq = faiss.ProductQuantizer(dim, 1, k)
    x_q = []
    for i in range(m):
        pq.train(data)
        codes = pq.compute_codes(data)

        if i == 0:
            rq_codes = codes
            datarec = pq.decode(codes)
        else:
            rq_codes = np.concatenate((rq_codes,codes),axis=1)
            datarec += pq.decode(codes)
        
        x_q.append(datarec.copy())
        data -= pq.decode(codes)
    return x_q, rq_codes

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()


class LabelSmoothingLoss(nn.Module):
    """
    标签平滑Loss
    """
    def __init__(self, classes, smoothing=0.0, dim=-1):
        """
        :param classes: 类别数目
        :param smoothing: 平滑系数
        :param dim: loss计算平均值的维度
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.loss = nn.KLDivLoss()
 
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        #torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
        loss = torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
        loss = self.loss(pred,true_dist)
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, alpha = 1, size_average = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001
    
    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        assert(logits.size(0) == labels.size(0))
        assert(logits.size(2) == labels.size(1))
        batch_size = logits.size(0)
        labels_length = logits.size(1)
        seq_length = logits.size(2)

        # transpose labels into labels onehot
        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([batch_size, labels_length, seq_length]).scatter_(1, new_label, 1)

        # calculate log
        log_p = F.log_softmax(logits)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt)**self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()
        
class InfoNCELoss(nn.Module):
    def __init__(self):
        super(InfoNCELoss, self).__init__()
        self.temp = 0.1
    
    def forward(self, output, target):
        # assert output.size()==target.size()
        output = F.normalize(output, dim=1)
        target = F.normalize(target, dim=1)

        d = 1 - torch.matmul(target, output.T)
        d = d / self.temp
        negexpd = torch.exp(-d)

        matrix_size = output.size(dim=0)
        labels = torch.eye(matrix_size, dtype=torch.float32).cuda()
        sacn = torch.sum(torch.multiply(negexpd, labels), dim=1)

        dmask = torch.ones([matrix_size, matrix_size], dtype=torch.float32).cuda()
        alcn = torch.sum(torch.multiply(negexpd, dmask), dim=1)
        
        loss = -torch.log((sacn)/alcn).mean()
        return loss
    
class ContrastiveTokenLoss(nn.Module):
    def __init__(self, pad_id=-1):
        super(ContrastiveTokenLoss, self).__init__()
        self.pad = pad_id
        
    def gen_neg(self, target, clssMap, clss):
        batch_size, seq_len = target.size()
        neg_tokens = torch.ones((batch_size, seq_len, self.num), dtype=torch.float32)
        # neg shape: [B, M, num_classes]

        for i in range(batch_size):
            target_cl = clss[i].item()
            maps_cls = clssMap[target_cl]
            for j in range(seq_len):
                pos = list(maps_cls.nodes.keys())
                for p in pos:
                    neg_tokens[i, j, p] = 0
                cur = target[i, j].item()
                maps_cls = maps_cls.nodes[cur]

        return neg_tokens
    
    def forward(self, input, target, clssMap, clss):
        #input [B, M, V]
        #target [B, M]
        self.num = input.size(-1)
        non_padding = target != self.pad #[B,M]
        neg_tokens = self.gen_neg(target, clssMap, clss).cuda()  #[B, M, V]
        

        positive_scores = input.gather(2, target.unsqueeze(-1)) #[B, M, 1]
        negative_scores = input * neg_tokens #[B, M, V]
        neg_minus_pos = negative_scores - positive_scores #[B, M, V]
        exp = neg_minus_pos.exp() #[B, M, V]

        sum_exp = (exp * neg_tokens).sum(dim=-1) #[B, M]
        losses = (1 + sum_exp).log() * non_padding.int() #[B, M]
        
        ct_loss = losses.sum() / non_padding.int().sum()
        return ct_loss
    

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.temp = 0.1
        
    def forward(self, output, clss, pred_clss, codebook, map_clid):
        batch_size = len(pred_clss)
        matrix_size = len(output)
        tok_size = int(matrix_size/batch_size)

        # mark same class predictions
        if torch.is_tensor(clss):
            clss = torch.flatten(clss)
            clss = np.array(clss.tolist())
        pred_clss = np.array(pred_clss)
        assert pred_clss.shape == clss.shape
        matches = clss==pred_clss

        target = []
        for i, (match, tgt_cls) in enumerate(zip(matches, clss)):
            if match:
                target.extend(output[i*tok_size:(i+1)*tok_size])
            else:
                # find nearest target with the src
                maps = map_clid[tgt_cls]
                tmp_tok = output[i*tok_size]
                for j in range(tok_size):
                    if tmp_tok in maps.nodes.keys():
                        target.append(output[i*tok_size+j])
                        maps = maps.nodes[tmp_tok]
                    else:
                        break
                    if j < tok_size-1:
                        tmp_tok = output[i*tok_size+j+1]
                        
                for k in range(j, tok_size):
                    tmp_tok = random.choice(list(maps.nodes))
                    target.append(tmp_tok)
                    maps = maps.nodes[tmp_tok]
        # print(len(target))
        assert len(target) == len(output)

        pred_emb = []
        target_emb = []
        for i, (pred_code, match_code) in enumerate(zip(output, target)):
            target_emb.append(codebook[i%4][match_code])
            pred_emb.append(codebook[i%4][pred_code])
        target_emb = torch.tensor(target_emb, requires_grad=True).float().cuda()
        target_emb = F.normalize(target_emb, dim=1)
        pred_emb = torch.tensor(pred_emb, requires_grad=True).float().cuda()
        pred_emb = F.normalize(pred_emb, dim=1)
        
        d = 1 - torch.matmul(target_emb, pred_emb.T)
        d = d / self.temp
        negexpd = torch.exp(-d)
        # print(negexpd.size())

        # same class labels
        labels = torch.zeros([matrix_size, matrix_size], dtype=torch.float32)
        tok_ones = torch.eye(tok_size, dtype=torch.float32)
        for i in range(batch_size):
            if matches[i]:
                labels[i*tok_size:(i+1)*tok_size, i*tok_size:(i+1)*tok_size] = tok_ones
        labels = labels.cuda()
        assert negexpd.size()==labels.size()
        sacn = torch.sum(torch.multiply(negexpd, labels), dim=1)
        
        # all class neighborhood
        dmask = torch.tile(tok_ones, (batch_size, batch_size))
        dmask = dmask.cuda()
        assert dmask.size()==labels.size()
        alcn = torch.sum(torch.multiply(negexpd, dmask), dim=1)
    
        # adding eps for numerical stability
        # in case of a class having a single occurance in batch
        # the quantity inside log would have been 0
        eps = 1e-9
        loss = -torch.log((sacn+eps)/alcn).mean()
        return loss