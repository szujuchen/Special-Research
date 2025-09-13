import networkx as nx 
import json
import pickle
import os
import torch
from torch_geometric.data import Data

file_dir = './processedData/smallmapping'
save_dir = './trainedData'
obj = json.load(open(os.path.join(file_dir, "mapping_playlist_song.json"), "r"))
G = nx.Graph()

node_arr = []
edge_arr = []
for item in obj:
    node_arr.append((item['playlist'], {'type' : 'playlist'}))
    for song in item['tracks']:
        node_arr.append((song, {'type': 'track'})) 
        edge_arr.append((item['playlist'], song))

G.add_nodes_from(node_arr)
G.add_edges_from(edge_arr)   
pickle.dump(G, open(os.path.join(save_dir, 'graph.txt'), 'wb'))

print("num nodes: ", G.number_of_nodes())
print("num edges: ", G.number_of_edges())

all_nodes = list(G.nodes())
all_edges = [list(e) for e in list(G.edges())]
all_edges.extend([list(e[::-1]) for e in list(G.edges())])
edge_idx = torch.LongTensor(all_edges)
data = Data(edge_index = edge_idx.t().contiguous(), num_nodes=G.number_of_nodes())
torch.save(data, os.path.join(save_dir, 'data_object.pt'))
stats = {'num_playlist': len(obj), 'num_nodes': G.number_of_nodes(), 'num_files': 30, 'num_edges': G.number_of_edges()}
with open(os.path.join(save_dir, 'stats.json'), 'w') as f:
    json.dump(stats, f)