# Read data/existing_episodes.json,
# create global dicts for date, uuid and path conversion.

from pathlib import Path
import json
import re
import numpy as np

imginfo = lambda img: print(type(img), img.dtype, img.shape, img.min(), img.max())

VISUAL = 0
MANUAL_ENABLE = 0
DROID_ROOT = Path(".")

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# subsample with replacement
def random_choice(a: np.ndarray, size: int) -> np.ndarray:
    # a has shape [n_vectors, ...]
    ind = np.random.randint(0, len(a), size=size)
    return a[ind]



# === existing episodes
with open(DROID_ROOT / "data/existing_episodes.json") as f:
    existing_episodes = json.load(f)
# 0 - date
# 1 - uuid
# 2 - path
# [ date "2023-03-02-15h-14m-31s", uuid "IRIS+ef107c48+2023-03-02-15h-14m-31s", path IRIS/success/(date)/(time) ]
uuid_to_remotepath = {episode[1]: episode[2] for episode in existing_episodes}
date_to_uuid = {episode[0]: episode[1] for episode in existing_episodes}


def convert_uuid_to_localpath(uuid: str):
    # uuid = IPRL+w026bb9b+2023-04-20-23h-28m-09s
    # remotepath = IPRL/success/2023-02-28/Tue_Feb_28_20:30:07_2023
    parts = uuid_to_remotepath[uuid].split("/")
    # localpath = data/droid_raw/1.0.1 / success/2023-02-28/Tue_Feb_28_20:30:07_2023
    localpath = Path("data/droid_raw/1.0.1/") / "/".join(parts[1:])
    return str(localpath)
date_to_localpath = {episode[0]: convert_uuid_to_localpath(episode[1]) for episode in existing_episodes}


# === annotations
_target_dir = DROID_ROOT / "data" / "droid_raw" / "1.0.1"
_annotations_file_name = "aggregated-annotations-030724.json"
annotations: dict[str, dict[str, str]]
with open(_target_dir / _annotations_file_name) as f:
    annotations = json.load(f)

# === saved episodes in awkward directory structure
saved_episodes: list[str] = []
for date in sorted((_target_dir / "success").iterdir()):
    for episode in sorted(date.iterdir()):
        # data/droid_raw/1.0.1/success/2023-03-02/Thu_Mar__2_15_00_02_2023
        # .    .         .     .       date       episode
        saved_episodes.append(str(episode))

def read_episode_date(episode: str):
    # read uuid
    json_file = list(Path(episode).glob("*json"))[0]
    with open(json_file, "r") as f:
        metadata = json.load(f)
    uuid = metadata["uuid"]

    # extract date
    regex = r'\w+\+\w+\+(\d+-\d+-\d+-\w+-\w+-\w+)$'
    date_str = re.findall(regex, uuid)[0]

    return date_str

# === manually selected episodes
manual_paths: list[str] = []
manual_dates: list[str] = []
train_idx: np.ndarray = None
val_idx: np.ndarray = None
if MANUAL_ENABLE:
    with open(DROID_ROOT / "data/manual_episodes.json", "r") as f:
        manual_dates = json.load(f)
        manual_paths = [date_to_localpath[date] for date in manual_dates]

    np.random.seed(0)
    _idx = np.random.permutation(len(manual_dates)) # 140
    train_idx = _idx[:120]
    val_idx = _idx[120:]

