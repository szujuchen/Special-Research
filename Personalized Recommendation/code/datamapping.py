import os
import json
import random
import numpy as np

data_dir = './spotifydata'
NUM_FILES_TO_USE = 30
save_dir = './processedData/smallmapping'

print("mapping given dataset...")
data_files = os.listdir(data_dir)
data_files = sorted(data_files, key=lambda x: int(x.split(".")[2].split("-")[0]))
data_files = data_files[:NUM_FILES_TO_USE]

playlist_set = {}
playlist_dict = {}
songs_set = {}
songs_dict = {}
mapping = []

for data_file in data_files:
    with open(os.path.join(data_dir, data_file), 'r') as f:
        d = json.load(f)['playlists']
        for playlist in d:
            playlist_info = {}
            playlist_info["ori_id"] = playlist["pid"]
            playlist_info["new_id"] = len(playlist_set)
            playlist_info["name"] = playlist["name"]
            playlist_set[playlist["pid"]] = playlist_info
            playlist_dict[playlist_info["new_id"]] = playlist["pid"]
            songs = []
            for track in playlist['tracks']:
                id = track['track_uri'].split(':')[-1]
                if id not in songs_set:
                    songs_info = {}
                    songs_info["ori_id"] = id
                    songs_info["new_id"] = len(songs_set) + NUM_FILES_TO_USE * 1000
                    songs_info["name"] = track["track_name"] 
                    songs_info["artist"] = track["artist_name"]
                    songs_info["album"] = track["album_name"]
                    songs_set[id] = songs_info
                    songs_dict[songs_info["new_id"]] = id
                
                temp = songs_set[id]
                songs.append(temp["new_id"])
            mapping.append({"playlist": playlist_info["new_id"], "tracks": songs})


print("original data stats:")
print("num of playlist: ", len(playlist_dict))
print("num of songs: ", len(songs_dict))
print("num of playlist with songs: ", len(mapping))

with open(os.path.join(save_dir, 'playlist_newtoold.json'), 'w') as f:
    json.dump(playlist_dict, f, indent=2)
with open(os.path.join(save_dir, 'playlist_oldtonew.json'), 'w') as f:
    json.dump(playlist_set, f, indent=2)
with open(os.path.join(save_dir, 'song_newtoold.json'), 'w') as f:
    json.dump(songs_dict, f, indent=2)
with open(os.path.join(save_dir, 'song_oldtonew.json'), 'w') as f:
    json.dump(songs_set, f, indent=2)
with open(os.path.join(save_dir, 'mapping_playlist_song.json'), 'w') as f:
    json.dump(mapping, f, indent=2)

# print("extracting training dataset...")
# train_item = 50000
# total = random.sample(list(np.arange(len(mapping))), train_item)
# trained = []
# extract the chosen playlist and track info
# unique_playlist = set()
# unique_song = set()
# for i in total:
#     item = mapping[i]
#     trained.append(item)
#     playlist = item["playlist"]
#     unique_playlist.add(playlist)
#     songs = item["tracks"]
#     for song in songs:
#         unique_song.add(song)

# playlists_item = []
# songs_item = []
# for playlist in unique_playlist:
#     old_id = playlist_dict[playlist]
#     info = playlist_set[old_id]
#     playlists_item.append(info)
# for song in unique_song:
#     old_id = songs_dict[song]
#     info = songs_set[old_id]
#     songs_item.append(info)
    
# print("training stats")
# print("num of training playlist: ", len(playlists_item))
# print("num of training songs: ", len(songs_item))

# test = random.sample(trained.copy(), len(trained) // 10)
# train = [item for item in trained if item not in test]
    
# print("writing to files")
# with open(os.path.join(save_dir, 'trained.json'), 'w') as f:
#     json.dump(trained, f, indent=2)
# with open(os.path.join(save_dir, 'train.json'), 'w') as f:
#     json.dump(train, f, indent=2)
# with open(os.path.join(save_dir, 'test.json'), 'w') as f:
#     json.dump(test, f, indent=2)
# with open(os.path.join(save_dir, 'trained_playlist.json'), 'w') as f:
#     json.dump(playlists_item, f, indent=2)
# with open(os.path.join(save_dir, 'trained_songs.json'), 'w') as f:
#     json.dump(songs_item, f, indent=2)



    
                