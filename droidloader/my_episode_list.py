# Read data/existing_episodes.json,
# create global dicts for conversion

from pathlib import Path
import json
import re

# === existing episodes
with open("data/existing_episodes.json") as f:
    existing_episodes = json.load(f)
# 0 - date
# 1 - uuid
# 2 - path
uuid_to_remotepath = {episode[1]: episode[2] for episode in existing_episodes}


def convert_uuid_to_localpath(uuid: str):
    # uuid = IPRL+w026bb9b+2023-04-20-23h-28m-09s
    # remotepath = IPRL/success/2023-02-28/Tue_Feb_28_20:30:07_2023
    parts = uuid_to_remotepath[uuid].split("/")
    # localpath = data/droid_raw/1.0.1 / success/2023-02-28/Tue_Feb_28_20:30:07_2023
    localpath = Path("data/droid_raw/1.0.1/") / "/".join(parts[1:])
    return str(localpath)
date_to_localpath = {episode[0]: convert_uuid_to_localpath(episode[1]) for episode in existing_episodes}


# === annotations
_target_dir = Path("data") / "droid_raw" / "1.0.1"
_annotations_file_name = "aggregated-annotations-030724.json"
annotations: dict[str, dict[str, str]]
with open(_target_dir / _annotations_file_name) as f:
    annotations = json.load(f)

# [ date "2023-03-02-15h-14m-31s", uuid "IRIS+ef107c48+2023-03-02-15h-14m-31s", path IRIS/success/(date)/(time) ]
# with open("data/existing_episodes.json") as f:
#     existing_episodes = json.load(f)

# list of available episodes
episodes: list[str] = []
for date in sorted((_target_dir / "success").iterdir()):
    for episode in sorted(date.iterdir()):
        # data/droid_raw/1.0.1/success/2023-03-02/Thu_Mar__2_15_00_02_2023
        # .    .         .     .       date       episode
        episodes.append(str(episode))
episodes = episodes[:10]

for i, episode in enumerate(episodes):
    print(f"{i: >4}", episode)
print("episodes:", len(episodes))


def read_episode_date(episode: str):
    # uuid of episode
    json_file = list(Path(episode).glob("*json"))[0]
    with open(json_file, "r") as f:
        metadata = json.load(f)
    uuid = metadata["uuid"]

    # extract date
    regex = r'\w+\+\w+\+(\d+-\d+-\d+-\w+-\w+-\w+)$'
    date_str = re.findall(regex, uuid)[0]

    return date_str
