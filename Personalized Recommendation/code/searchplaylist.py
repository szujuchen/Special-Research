from elasticsearch import Elasticsearch
from elasticsearch.helpers import parallel_bulk
import json
import os
import tqdm
from argparse import ArgumentParser
import secrets

parser = ArgumentParser()
parser.add_argument("-k", "--keyword", dest="keyword", default="")
args = parser.parse_args()

path_to_ca_cert = "/tmp2/b10705005/elasticsearch-8.13.4/config/certs/http_ca.crt"

client = Elasticsearch(
  "https://localhost:9200",
  ca_certs=path_to_ca_cert,
  basic_auth=("elastic", secrets.elastic_password),
  request_timeout=60,
  max_retries=3,
  retry_on_timeout=True,
)

def searchplaylist(keyword):
    if(keyword == ""):
        return []
    
    q = {
        "size": 10,
        "query": {  
            "multi_match":{
                "query": keyword,
                "fields": ["name"]
            },
        },      
        "_source": ["new_id"]
    }
    
    res = client.search(index="playlistdataset", body=q)
    if res["hits"]["total"]["value"] == 0:
        return []
    results = []
    for item in res["hits"]["hits"]:
        results.append(item["_source"]["new_id"])
    return results

# print(searchplaylist(args.keyword))