import json
import os
import csv

data_dir = "./processedData/smallmapping"
header = ["user", "item", "label"]
data = json.load(open(os.path.join(data_dir, "mapping_playlist_song.json"), "r"))
datas = []
for item in data:
    for track in item["tracks"]:
        datas.append([item["playlist"], track, 1])
        
with open(os.path.join(data_dir, 'mapping.csv'), 'w') as f:
    write = csv.writer(f)
    # write.writerow(header)
    write.writerows(datas)