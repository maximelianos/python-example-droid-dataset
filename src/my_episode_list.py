# Read data/existing_episodes.json,
# create global dicts for conversion

from pathlib import Path
import json

# === existing episodes
with open("data/existing_episodes.json") as f:
    existing_episodes = json.load(f)
# 0 - date
# 1 - uuid
# 2 - path
uuid_to_path = {episode[1]: episode[2] for episode in existing_episodes}


def convert_uuid_to_localpath(uuid: str):
    # uuid = IPRL+w026bb9b+2023-04-20-23h-28m-09s
    # remotepath = IPRL/success/2023-02-28/Tue_Feb_28_20:30:07_2023
    parts = uuid_to_path[uuid].split("/")
    # localpath = data/droid_raw/1.0.1 / success/2023-02-28/Tue_Feb_28_20:30:07_2023
    localpath = Path("data/droid_raw/1.0.1/") / "/".join(parts[1:])
    return str(localpath)

date_to_localpath = {episode[0]: convert_uuid_to_localpath(episode[1]) for episode in existing_episodes}
