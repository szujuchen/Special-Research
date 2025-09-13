from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import secrets

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=secrets.spotify_cid, client_secret=secrets.spotify_secret))
# results = sp.search(q='ska+year%3A2010-2017', limit=20, type="track")
# print(results)
# for idx, track in enumerate(results['tracks']['items']):
#     print(idx, track['name'])
    
def search_ori(keyword):
    if keyword == "":
        return []
    recommend = []
    results = sp.search(q=f'{keyword}+year%3A2010-2017' , limit=10, type="track")
    # print(results)
    for res in results["tracks"]["items"]:
        recommend.append({"id": res['id'], "title": res['name'], "artist": res['artists'][0]['name']})
    
    # print(recommend)
    return recommend
    
# print(search_ori("relax"))