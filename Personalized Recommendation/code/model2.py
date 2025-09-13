import numpy as np
import os
import pandas as pd
import torch
from libreco.data import random_split, DatasetPure
from libreco.algorithms import LightGCN
from libreco.evaluation import evaluate

data_dir = './processedData/smallmapping'
save_dir = './trainedData/libreco100epoch'
data = pd.read_csv(os.path.join(data_dir, "mapping.csv"), sep=",",
                   names=["user", "item", "label"])

# split whole data into three folds for training, evaluating and testing
train_data, eval_data, test_data = random_split(data, multi_ratios=[0.8, 0.1, 0.1])

train_data, data_info = DatasetPure.build_trainset(train_data)
eval_data = DatasetPure.build_evalset(eval_data)
test_data = DatasetPure.build_testset(test_data)
print(data_info)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
lightgcn = LightGCN(
    task="ranking",
    data_info=data_info,
    loss_type="bpr",
    embed_size=64,
    n_epochs=100,
    lr=1e-3,
    batch_size=2048,
    sampler="random",
    num_neg=1,
    seed=5,
    device=device,
)
# monitor metrics on eval data during training
lightgcn.fit(
    train_data,
    neg_sampling=True,
    verbose=2,
    eval_data=eval_data,
    metrics=["loss", "roc_auc", "precision", "recall", "ndcg"],
    k=300,
)

# do final evaluation on test data
print("evaluation", 
    evaluate(
        model=lightgcn,
        data=test_data,
        neg_sampling=True,
        metrics=["loss", "roc_auc", "precision", "recall", "ndcg"],
    )
)


if not os.path.exists(save_dir):
    os.mkdir(save_dir)

data_info.save(path=save_dir, model_name="lightgcn_data")
lightgcn.save(path=save_dir, model_name="lightgcn_model", manual=True, inference_only=True)