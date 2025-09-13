from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import json
import os
import tqdm
import secrets

path_to_ca_cert = "/tmp2/b10705005/elasticsearch-8.13.4/config/certs/http_ca.crt"

client = Elasticsearch(
  "https://localhost:9200",
  ca_certs=path_to_ca_cert,
  basic_auth=("elastic", secrets.elastic_password),
  request_timeout=60,
  max_retries=3,
  retry_on_timeout=True,
)
# print(client.info())

# create index
# mappings = {
#     "properties": {
#         "ori_id": {"type": "keyword"},
#         "new_id": {"type": "integer"},
#         "name": {"type": "text"},
#         "artist": {"type": "text"},
#         "album": {"type": "text"},
#         "lyrics": {"type": "text"},
#         "type": {"type": "text"},
#     }
# }
# client.indices.create(index="songdataset", mappings=mappings)
# print("song index created")


# data_dir = './processedData/smallmapping'
# data0 = json.load(open(os.path.join(data_dir, "song_info0.json")))
# data1 = json.load(open(os.path.join(data_dir, "song_info1.json")))
# data2 = json.load(open(os.path.join(data_dir, "song_info2.json")))
# data3 = json.load(open(os.path.join(data_dir, "song_info3.json")))

# bulk_data = []
# for data in data0.values():
#     data["type"] = "track"
#     bulk_data.append({
#         "_index": "songdataset",
#         "_id": data["new_id"],
#         "_source": data
#     })
# for data in data1.values():
#     data["type"] = "track"
#     bulk_data.append({
#         "_index": "songdataset",
#         "_id": data["new_id"],
#         "_source": data
#     })
# for data in data2.values():
#     data["type"] = "track"
#     bulk_data.append({
#         "_index": "songdataset",
#         "_id": data["new_id"],
#         "_source": data
#     })
# for data in data3.values():
#     data["type"] = "track"
#     bulk_data.append({
#         "_index": "songdataset",
#         "_id": data["new_id"],
#         "_source": data
#     })
# print("finish list appending")
# print(len(data0) + len(data1) + len(data2) + len(data3))
# bulk(client, bulk_data)
    

# #create index
mappings = {
    "properties": {
        "ori_id": {"type": "keyword"},
        "new_id": {"type": "integer"},
        "name": {"type": "text"},
    }
}
client.indices.create(index="playlistdataset", mappings=mappings)
print("playlist index created")

bulk_data = []
data_dir = './processedData/smallmapping'
datas = json.load(open(os.path.join(data_dir, "playlist_oldtonew.json")))
for data in datas.values():
    data["type"] = "playlist"
    bulk_data.append({
        "_index": "playlistdataset",
        "_id": data["new_id"],
        "_source": data
    })
print("playlist finish appending")
print(len(datas))
bulk(client, bulk_data)