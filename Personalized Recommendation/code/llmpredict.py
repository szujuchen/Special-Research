# get extended candidate from semantic 
from openai import OpenAI
import json
from searchsong import searchsong
import secrets

client = OpenAI(
    api_key=secrets.openai_api,
)
instruction = "You are an excellent assistant great at generating relavant keywords.\nNow youare going to help the user generate 10 relative keywords to search for the music base on the instruction user gives you.\nPlease return the keywords as a JSON object like this: {\"keywords\":[]}."

# extend keyword
def llmextend(keyword):
    if keyword == "":
        keyword = "just give me nine keywords to search for music"
    else:
        keyword = f"just give me nine kewords to search for music based on the user input keyword: {keyword}"
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": instruction},
        {"role": "user", "content": keyword}
    ]
    )
    return json.loads(completion.choices[0].message.content)["keywords"]

# generate keyword based on history
def llmgen(num, history):
    keyword = f"Give me {num} keywords to search for the next music based on my listen history: {history}. Remember only {num} keywords."
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": instruction},
        {"role": "user", "content": keyword}
    ]
    )
    return json.loads(completion.choices[0].message.content)["keywords"]

# combine old keyword and new keyword
def llmrel(oldext, newext):
    keyword = f"Here is the old keyword extension: {oldext}. And the new keyword extension based on listening history: {newext}. Please regenerate ten relavant keywords based on the above two keyword extensions set. Remeber only ten keywords."
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": instruction},
        {"role": "user", "content": keyword}
    ]
    )
    return json.loads(completion.choices[0].message.content)["keywords"]

# generate keyword and search result for user search
def cand_gen(keyword):
    extension = llmextend(keyword)
    extension.insert(0, keyword)
    # print(extension)
    curkey = []
    cand = []
    for ext in extension:
        res = searchsong(ext)
        res.sort(reverse=True)
        cand.extend(res)
        if ext == keyword:
            curkey = res
    # sort them with score relavant to keyword
    cand.sort(reverse=True)
    curkey = [song[1] for song in curkey]
    
    once = []
    repl = []
    for item in cand:
        if item[1] not in once:
            once.append(item[1])
        else:
            repl.append(item[1])
    final = list(set(once) - set(repl))
    return curkey, repl, final, extension

# generate keyword and recommendations of clicked somes
def rela_gen(extkey, history):
    # print("old: ", extkey)
    history_word = "\n"
    for i, item in enumerate(history):
        history_word += f"{i+1}. title: {item['name']}, artist: {item['artist']}, album: {item['album']}"
        history_word += "\n"
    # generate new keyword for watching history
    cur = llmgen(1, history_word)
    # generate new keyword for spotify search
    spotifykey = cur[0]
    # combine the old and new ones for next recommendation
    extension = llmrel(extkey, cur)
    # print("combined: ", extkey)
    print("spotify: ", spotifykey)
    cur = cur[0]
    curkey = []
    cand = []
    for ext in extension:
        res = searchsong(ext)
        res.sort(reverse=True)
        cand.extend(res)
        if ext == cur:
            curkey = res
        
    if cur not in extension:
        res = searchsong(cur)
        res.sort(reverse=True)
        cand.extend(res)
        curkey = res
        
    cand.sort(reverse=True)
    cand = [song[1] for song in cand]
    curkey = [song[1] for song in curkey]
    
    once = []
    repl = []
    for item in cand:
        if item not in once:
            once.append(item)
        else:
            repl.append(item)
    final = list(set(once) - set(repl))
    return cand, curkey, repl, final, extension, spotifykey


# res = cand_gen("relax")
# print(res)
# print("candidate num: ", len(res))