import os
import json
import pickle
import networkx as nx
import deepsnap
from deepsnap.graph import Graph
from deepsnap.dataset import GraphDataset

file_dir = './trainedData'
G = pickle.load(open(os.path.join(file_dir, 'graph.txt'), 'rb'))

nodes = [id for id, attr in G.nodes(data=True)]
edges = [edge for edge in G.edges()]

G_temp = nx.Graph()
G_temp.add_nodes_from(nodes)
G_temp.add_edges_from(edges)

dg = Graph(G_temp)
task = 'link_pred'
dataset = GraphDataset([dg], task=task, edge_train_mode='disjoint')
train, val, test = dataset.split(transductive=True, split_ratio=[0.8, 0.1, 0.1])

num_train_edges = train[0].edge_label_index.shape[1]
num_val_edges = val[0].edge_label_index.shape[1]
num_test_edges = test[0].edge_label_index.shape[1]
stats = {}
stats["train"] = {"supervision(positive)" : num_train_edges // 2, "message passing": train[0].edge_index.shape[1]}
stats["val"] = {"supervision(positive)" : num_val_edges // 2, "message passing": val[0].edge_index.shape[1]}
stats["test"] = {"supervision(positive)" : num_test_edges // 2, "message passing": test[0].edge_index.shape[1]}

print(stats)
with open(os.path.join(file_dir, 'stats.json'), 'w') as f:
    json.dump(stats, f, indent=2)

pickle.dump(train, open(os.path.join(file_dir, 'train.graph'), 'wb'))
pickle.dump(val, open(os.path.join(file_dir, 'val.graph'), 'wb'))
pickle.dump(test, open(os.path.join(file_dir, 'test.graph'), 'wb'))

