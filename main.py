
import time
from typing import Counter
from riotwatcher import LolWatcher, ApiError
import pandas as pd

api_key = "RGAPI-6cce9a9f-6d28-4851-b141-621b492dd041"
watcher = LolWatcher(api_key)
region = "euw1"


#Get information of players String
player = watcher.summoner.by_name(region, "BarbeQ")

ranked_status = watcher.league.by_summoner(region, player["id"])



puuid = player["puuid"]

#If want more than 10 specify number with variable count
match = watcher.match.matchlist_by_puuid("europe", puuid= puuid, count= 100)

data = []
for game in match:
    match_detail = watcher.match.by_id("europe", game)

    info = list(match_detail.keys())[1]
    #print(match_detail[info]["participants"][9])

    
    version = watcher.data_dragon.versions_for_region(region)

    for cosa in match_detail[info]["participants"]:
        if cosa["summonerName"] == "BarbeQ":
            rows = {}
            rows["Champion_Name"] = cosa["championName"]
            rows["Username"] = cosa["summonerName"]
            rows["Kills"] = cosa["kills"]
            rows["Deaths"] = cosa["deaths"]
            rows["Assists"] = cosa["assists"]
            rows["Farm"] = cosa["totalMinionsKilled"]
            rows["Gold_Earn"] = cosa["goldEarned"]
            if cosa["teamPosition"] == "UTILITY":
                rows["Position"] = "Support"
            else:
                rows["Position"] = cosa["teamPosition"].title()
    
            rows["Total_Damage"] = cosa["totalDamageDealtToChampions"]
            rows["Level"] = cosa["champLevel"]
            rows["Win"] = cosa["win"]
            data.append(rows)
    
df = pd.DataFrame(data)
print(df)   

df.to_csv("Datos_User.csv", index= False)

