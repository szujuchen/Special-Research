from llmpredict import cand_gen, rela_gen
from modelpredict import behavior_cand_gen, clickrec_gen, searchrec_gen
import os, json
from datetime import datetime
import random

data_dir = "./processedData/smallmapping"
songs_data = json.load(open(os.path.join(data_dir, "song_oldtonew.json"), "r"))
newid_data = json.load(open(os.path.join(data_dir, "song_newtoold.json"), "r"))

def final_recommend(keyword):
    semantics, ext = cand_gen(keyword)
    behaviors, history = behavior_cand_gen(keyword)
    # intersections
    final = list(set(semantics) & set(behaviors))
    
    # get info for candidate
    data_dir = "./processedData/smallmapping"
    songs_data = json.load(open(os.path.join(data_dir, "song_newtoold.json"), "r"))
    semantics_ids = []
    behaviors_ids = []
    final_ids = []
    if final:
        for item in final:
            final_ids.append(f"https://open.spotify.com/embed/track/{songs_data[str(item)]}?utm_source=generator&theme=0")
    for i in range(10):
        semantics_ids.append(f"https://open.spotify.com/embed/track/{songs_data[str(semantics[i])]}?utm_source=generator&theme=0")
    for i in range(10):
        behaviors_ids.append(f"https://open.spotify.com/embed/track/{songs_data[str(behaviors[i])]}?utm_source=generator&theme=0")
    # writing to history json file
    history_dir = "./trainedData/"
    historys = None
    if os.path.exists(os.path.join(history_dir, "history.json")):
        historys = json.load(open(os.path.join(history_dir, "history.json"), "r"))
    if history is not None:
        if final:
            historys[history]["tracks"].append(final[0])
        else:
            historys[history]["tracks"].append(semantics[0])
            historys[history]["tracks"].append(behaviors[0])
        historys[history]["modifiedTime"] = str(datetime.now())
        keys = set(historys[history]["keywords"])
        keys.add(keyword)
        historys[history]["keywords"] = list(keys)
        newHistory = sorted(historys, key=lambda h: h['modifiedTime'])
    else:
        temphistory = {}
        if final:
            temphistory["tracks"] = [final[0]]
        else:
            temphistory["tracks"] = [semantics[0], behaviors[0]]
        temphistory["keywords"] = [keyword]
        temphistory["modifiedTime"] = str(datetime.now())
        if historys:
            historys.append(temphistory)
            newHistory = historys
        else:
            newHistory = []
            newHistory.append(temphistory)
    with open(os.path.join(history_dir, "history.json"), "w") as f:
        json.dump(newHistory, f, indent=2)
        
    return final_ids, semantics_ids, behaviors_ids, ext
    
def search_llm(keyword):
    llm_display, repeat, once, extkey = cand_gen(keyword)
    # print("llm: ", llm_display)
    # print("repeat: ", repeat)
    # print("all search: ", all)
    # print("keys: ", extkey)
    gcn_display = searchrec_gen(repeat, once, llm_display)
    # print("gcn: ", gcn_display)
    
    ids = []
    # 5 for llm 5 for gcn # half half for not yet user hisotry
    for i in range(5):
        id = newid_data[str(llm_display[i])]
        ids.append({"id":id, "title": songs_data[id]["name"], "artist": songs_data[id]["artist"], "type": "llm"})
        id = newid_data[str(gcn_display[i])]
        ids.append({"id":id, "title": songs_data[id]["name"], "artist": songs_data[id]["artist"], "type": "gcn"})
    
    shuffled = sorted(ids, key=lambda x: random.random())
    return shuffled, extkey

def search_rela(extkey, history, llmcnt):
    clicked = []
    history_ids = []
    for item in history:
        clicked.append(songs_data[item])
        history_ids.append(songs_data[item]["new_id"])
        
    llm_display, history_rec, repeat, once, extkey, spotifykey = rela_gen(extkey, clicked)
    gcn_display = clickrec_gen(history_ids, repeat, once, history_rec)
    
    final_ids = []
    for i in range(llmcnt):
        id = newid_data[str(llm_display[i])]
        final_ids.append({"id":id, "title": songs_data[id]["name"], "artist": songs_data[id]["artist"], "type": "llm"})
    for i in range(10 - llmcnt):
        id = newid_data[str(gcn_display[i])]
        final_ids.append({"id":id, "title": songs_data[id]["name"], "artist": songs_data[id]["artist"], "type": "gcn"})
    
    shuffled = sorted(final_ids, key=lambda x: random.random())
    return shuffled, extkey, spotifykey