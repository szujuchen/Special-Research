import os
import json
from lyricsgenius import Genius
from tqdm import tqdm
from multiprocessing import Pool, Manager
import secrets

data_dir = './processedData/smallmapping'

genius = Genius(secrets.genius_token)
genius.verbose = False
genius.remove_section_headers = True
genius.skip_non_songs = False
genius.excluded_terms = ["(Remix)", "(Live)"]
genius.timeout = 15
genius.retries = 3

datas = json.load(open(os.path.join(data_dir, "song_oldtonew.json"), "r"))
keys = list(datas.keys())
assert len(keys) == len(datas)
all = Manager().dict()

def get_lyrics(data):
    song = None
    try:
        song = genius.search_song(data["name"],data["artist"])
    except:
        pass
    if song is None:
        data["lyrics"] =  ""
    else:
        data["lyrics"] =  song.lyrics
    all[data["ori_id"]] = data
    return

with Pool(10) as pool:
    for i in tqdm(range(len(keys))):
        pool.apply_async(get_lyrics, args=(datas[keys[i]], ), error_callback=lambda e: print(e))
    pool.close()
    pool.join()
       
with open(os.path.join(data_dir, "song_info.json"), "w") as f:
    json.dump(all.copy(), f, indent=2)
        