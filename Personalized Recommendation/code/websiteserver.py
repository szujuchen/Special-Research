from flask import Flask, render_template, request
from predict import search_llm, search_rela
from spotifysearch import search_ori
import json 
import os

app = Flask(__name__)
global keyword
keyword = None
global extkey
extkey = None
global history
history = []
global llmcnt
llmcnt = 5

data_dir = "./processedData/smallmapping"
songs_data = json.load(open(os.path.join(data_dir, "song_oldtonew.json"), "r"))

@app.route('/')
def index():
  output = request.form.to_dict()
  print(output)
  return render_template("index.html")

# @app.route('/result',methods=['POST', 'GET'])
# def result():
#     output = request.form.to_dict()
#     print(output)
#     keyword = output["keyword"]
#     union, semantic, behavior, ext = final_recommend(keyword)
#     spotify = search_ori(keyword)
#     return render_template('index.html', keyword = keyword, extend = ext, union = union, semantic = semantic, behavior = behavior, spotify = spotify, search=True)
  
@app.route('/search',methods=['POST', 'GET'])
def search():
  global keyword
  global history
  global extkey
  global llmcnt
  history = []
  llmcnt = 5
  output = request.form.to_dict()
  # print(output)
  keyword = output["keyword"]
  recs, extkey = search_llm(keyword)
  # compared with original search result
  spotify = search_ori(keyword)
  return render_template('index.html', type="search", keyword=keyword, extend=extkey, display=recs, spotify=spotify)
  
@app.route('/click/<type>/<track_id>',methods=['POST', 'GET'])
def click(type, track_id):
  global keyword
  global history
  global extkey
  global llmcnt
  # preference based on user behavior
  if type == "llm" and llmcnt < 8:
    llmcnt += 1
  elif type == "gcn" and llmcnt > 2:
    llmcnt -= 1
  
  cur_track = {"id":track_id, "title": songs_data[track_id]["name"], "artist": songs_data[track_id]["artist"]}
  history.append(track_id)
  recs, extkey, spotifykey = search_rela(extkey, history, llmcnt)
  # compared with original search results
  spotify = search_ori(spotifykey)
  return render_template('index.html', type="click", track=cur_track, keyword=keyword, extend=extkey, display=recs, spotify=spotify, spotifykey=spotifykey)

if __name__ == '__main__':
  app.run(host="0.0.0.0", port=8000)