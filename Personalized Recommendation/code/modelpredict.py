import os
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from libreco.algorithms import LightGCN
from libreco.data import DataInfo
# from argparse import ArgumentParser
from searchplaylist import searchplaylist

# parser = ArgumentParser()
# parser.add_argument("-k", "--keyword", dest="keyword", default="")
# args = parser.parse_args()

def filter_items(ids, preds, items):
    mask = np.isin(ids, items, assume_unique=True, invert=True)
    return ids[mask], preds[mask]

def partition_select(ids, preds, n_rec):
    mask = np.argpartition(preds, -n_rec)[-n_rec:]
    return ids[mask], preds[mask]

def behavior_cand_gen(keyword):
    #load model
    tf.compat.v1.reset_default_graph()
    save_dir = "./trainedData/libreco100epoch"
    data_info = DataInfo.load(save_dir, model_name="lightgcn_data")
    model = LightGCN.load(path=save_dir, model_name="lightgcn_model", data_info=data_info, manual=True)
    item_emb = model.get_item_embedding()[: model.n_items]

    # keyword = args.keyword.lower()
    K = 10
    num_playlists = model.n_users
    # find matching playlist that has throwned into train
    data_dir = "./processedData/smallmapping"
    temp_playlist = searchplaylist(keyword)
    # print("relavant playlist: ", temp_playlist)
    # find matching history
    history_dir = "./trainedData/"
    relavantHistory = None
    historys = None
    heared = []
    temp_song = []
    if os.path.exists(os.path.join(history_dir, "history.json")):
        historys = json.load(open(os.path.join(history_dir, "history.json"), "r"))
        found = False
        if keyword == "":
            relavantHistory = len(historys) - 1
            heared = historys[-1]["tracks"]
            temp_song = historys[-1]["tracks"]
        else:
            for id, history in enumerate(historys[::-1]):
                for key in history["keywords"]:
                    if keyword in key.lower() or key.lower() in keyword:
                        found = True
                        break
                if found:
                    relavantHistory = len(historys) - id - 1
                    heared = history["tracks"]
                    temp_song = history["tracks"]
                    break
                
    heared = [num - num_playlists for num in heared]
    # print("history: ", relavantHistory)
    # print("history tracks: ", heared)

    # get embeddings from history records            
    cand_playlist = None
    cand_song = None
    if temp_playlist:
        temp_emb = []
        for temp in temp_playlist:
            try: 
                temp_emb.append(model.get_user_embedding(temp))
            except:
                pass
        cand_playlist = np.mean(np.array(temp_emb), axis=0)
    if temp_song:
        temp_emb = []
        for temp in temp_song:
            try:
                temp_emb.append(model.get_item_embedding(temp))
            except:
                pass
        cand_song = np.mean(np.array(temp_emb), axis=0)

    emb_key = None
    if cand_playlist is not None and cand_song is not None:
        emb_key = 0.5*cand_playlist + 0.5*cand_song
    elif cand_playlist is not None:
        emb_key = cand_playlist
    elif cand_song is not None:
        emb_key = cand_song
    # print(emb_key)
        
    # have some reference data to recommend
    if emb_key is not None:
        preds = emb_key @ item_emb.T
        if preds.ndim == 1:
            assert len(preds) % model.n_items == 0
            batch_size = int(len(preds) / model.n_items)
            all_preds = preds.reshape(batch_size, model.n_items)
        else:
            batch_size = len(preds)
            all_preds = preds
        all_ids = np.tile(np.arange(model.n_items), (batch_size, 1))

        batch_ids, batch_preds = [], []
        for i in range(batch_size):
            ids = all_ids[i]
            preds = all_preds[i]
            
            if len(heared) > 0 and K + len(heared) <= model.n_items:
                ids, preds = filter_items(ids, preds, heared)
        
            ids, preds = partition_select(ids, preds, K)
            batch_ids.append(ids)
            batch_preds.append(preds)

        ids, preds = np.array(batch_ids), np.array(batch_preds)
        indices = np.argsort(preds, axis=1)[:, ::-1]
        ids = np.take_along_axis(ids, indices, axis=1)[0]
    else:
        recommendation = model.recommend_user(user=model.n_users+1, n_rec=K, cold_start="popular")
        ids = recommendation[model.n_users+1].to_list()

    recs = [int(id + num_playlists) for id in ids]
    return recs, relavantHistory

# if relavantHistory is not None:
#     historys[relavantHistory]["tracks"].append(recs[0])
#     historys[relavantHistory]["modifiedTime"] = str(datetime.now())
#     keys = set(historys[relavantHistory]["keywords"])
#     keys.add(keyword)
#     historys[relavantHistory]["keywords"] = list(keys)
#     newHistory = sorted(historys, key=lambda h: h['modifiedTime'])
# else:
#     temphistory = {}
#     temphistory["tracks"] = [recs[0]]
#     temphistory["keywords"] = [keyword]
#     temphistory["modifiedTime"] = str(datetime.now())
#     if historys:
#         historys.append(temphistory)
#         newHistory = historys
#     else:
#         newHistory = []
#         newHistory.append(temphistory)
# with open(os.path.join(history_dir, "history.json"), "w") as f:
#     json.dump(newHistory, f, indent=2)
    
# print("recommendations: ", behavior_cand_gen("relax"))

def history_gen(history, rela):
    #load model
    tf.compat.v1.reset_default_graph()
    save_dir = "./trainedData/libreco100epoch"
    data_info = DataInfo.load(save_dir, model_name="lightgcn_data")
    model = LightGCN.load(path=save_dir, model_name="lightgcn_model", data_info=data_info, manual=True)
    item_emb = model.get_item_embedding()[: model.n_items]
    K = 10
    num_playlists = model.n_users
    relative = [num - num_playlists for num in rela]
    heared = [num - num_playlists for num in history]
    emb = []
    for song in relative:
        try:
            emb.append(model.get_item_embedding(song))
        except:
            pass
    for song in heared:
        try:
            emb.append(model.get_item_embedding(song))
        except:
            pass
    final_emb = np.mean(np.array(emb), axis=0)
    
    preds = final_emb @ item_emb.T
    if preds.ndim == 1:
        assert len(preds) % model.n_items == 0
        batch_size = int(len(preds) / model.n_items)
        all_preds = preds.reshape(batch_size, model.n_items)
    else:
        batch_size = len(preds)
        all_preds = preds
    all_ids = np.tile(np.arange(model.n_items), (batch_size, 1))

    batch_ids, batch_preds = [], []
    for i in range(batch_size):
        ids = all_ids[i]
        preds = all_preds[i]
        
        if len(heared) > 0 and K + len(heared) <= model.n_items:
            ids, preds = filter_items(ids, preds, heared)
    
        ids, preds = partition_select(ids, preds, K)
        batch_ids.append(ids)
        batch_preds.append(preds)

    ids, preds = np.array(batch_ids), np.array(batch_preds)
    indices = np.argsort(preds, axis=1)[:, ::-1]
    ids = np.take_along_axis(ids, indices, axis=1)[0]
    recs = [int(id + num_playlists) for id in ids]
    return recs

# search result generation
def searchrec_gen(repeat, results, llmrec):
    #load model
    tf.compat.v1.reset_default_graph()
    save_dir = "./trainedData/libreco100epoch"
    data_info = DataInfo.load(save_dir, model_name="lightgcn_data")
    model = LightGCN.load(path=save_dir, model_name="lightgcn_model", data_info=data_info, manual=True)
    item_emb = model.get_item_embedding()[: model.n_items]
    K = 10
    num_playlists = model.n_users
    heared = []
    all_rec = []
    emb = []
    # emb = 1.2 * repeat + 2 * ori_keyword + other search result
    # emphasize on repeat item in llm search (semantic)
    if len(repeat) > 0:
        relative = [num - num_playlists for num in repeat]
        for song in relative:
            try:
                emb.append(1.2 * model.get_item_embedding(song))
            except:
                pass
    # the user keyword search result have higher weight in embedding
    user = [num - num_playlists for num in llmrec]
    for song in user:
        try:
            emb.append(model.get_item_embedding(song))
        except:
            pass
    # add other keyword result embedding
    relative = [num - num_playlists for num in results]
    for song in relative:
        try:
            emb.append(model.get_item_embedding(song))
        except:
            pass
        
    all_rec.extend(cal_rec(emb, item_emb, model, K, heared, num_playlists))
    # sort them with keyword relavant score
    # return the top 10
    all_rec.sort(reverse=True)
    recommendation = []
    for item in all_rec:
        if item[1] not in recommendation:
            recommendation.append(item[1])
    return recommendation


def clickrec_gen(history, repeat, results, llmrec):
    #load model
    tf.compat.v1.reset_default_graph()
    save_dir = "./trainedData/libreco100epoch"
    data_info = DataInfo.load(save_dir, model_name="lightgcn_data")
    model = LightGCN.load(path=save_dir, model_name="lightgcn_model", data_info=data_info, manual=True)
    item_emb = model.get_item_embedding()[: model.n_items]
    K = 10
    num_playlists = model.n_users
    heared = [num - num_playlists for num in history]
    all_rec = []
    # perform behavior on each extended keyword
    emb = []
    # emb = 1.2 * repeat + 1.5 * relative search results to new key + 2(series) * user history + other search result
    # emphasize on repeat item in llm search (semantic)
    if len(repeat) > 0:
        relative = [num - num_playlists for num in repeat]
        for song in relative:
            try:
                emb.append(1.2 * model.get_item_embedding(song))
            except:
                pass
    # the history have higher weights
    user = [num - num_playlists for num in history]
    weights = np.logspace(start=1, stop=2, base=2, num=len(user))
    for ind, song in enumerate(user):
        try:
            emb.append(weights[ind] * model.get_item_embedding(song))
        except:
            pass
    # the search result based on only history keyword
    relative = [num - num_playlists for num in llmrec]
    for ind, song in enumerate(user):
        try:
            emb.append(0.5 * model.get_item_embedding(song))
        except:
            pass
    # add other keyword result embedding
    relative = [num - num_playlists for num in results]
    for song in relative:
        try:
            emb.append(model.get_item_embedding(song))
        except:
            pass 
        
    all_rec.extend(cal_rec(emb, item_emb, model, K, heared, num_playlists))
    # sort them with keyword relavant score
    # return the top 10
    all_rec.sort(reverse=True)
    recommendation = []
    for item in all_rec:
        if item[1] not in recommendation:
            recommendation.append(item[1])
    return recommendation
        

def cal_rec(emb, item_emb, model, K, heared, num_playlists):
    final_emb = np.mean(np.array(emb), axis=0)
    preds = final_emb @ item_emb.T
    if preds.ndim == 1:
        assert len(preds) % model.n_items == 0
        batch_size = int(len(preds) / model.n_items)
        all_preds = preds.reshape(batch_size, model.n_items)
    else:
        batch_size = len(preds)
        all_preds = preds
    all_ids = np.tile(np.arange(model.n_items), (batch_size, 1))

    batch_ids, batch_preds = [], []
    for i in range(batch_size):
        ids = all_ids[i]
        preds = all_preds[i]
        
        if len(heared) > 0 and K + len(heared) <= model.n_items:
            ids, preds = filter_items(ids, preds, heared)
    
        ids, preds = partition_select(ids, preds, K)
        batch_ids.append(ids)
        batch_preds.append(preds)

    ids, preds = np.array(batch_ids), np.array(batch_preds)
    indices = np.argsort(preds, axis=1)[:, ::-1]
    ids = np.take_along_axis(ids, indices, axis=1)[0]
    score = np.take_along_axis(preds, indices, axis=1)[0]
    recs = [(score[index], int(id + num_playlists)) for index, id in enumerate(ids)]
    return recs