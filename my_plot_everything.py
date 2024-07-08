from pathlib import Path
import json
import re
import datetime as dt
import subprocess


root_dir = Path(__file__).parent
target_dir = root_dir / "data" / "droid_raw" / "1.0.1"
annotations_file_name = "aggregated-annotations-030724.json"
with open(target_dir / annotations_file_name) as f:
    annotations = json.load(f)

# === result scheme
# === { "IRIS+ef107c48+2023-03-02-15h-14m-31s": IRIS/success/(date)/(time) }

data = Path("data/droid_raw/1.0.1/success")
episodes = []
for date in sorted(data.iterdir()):
    for episode in sorted(date.iterdir()):
        episodes.append(episode)
        print(f"{len(episodes): >4}", episode)
# INTER
input("continue...")

"""
log structure:
{
    "uuid": "IRIS+ef107c48+2023-03-02-15h-14m-31s"
    "org": "IRIS"
    "day": "2023-07-07"
    "visible_points": int
    "gripper_closed_times": int
    "gripper_duration": seconds
    "episode_duration": seconds
}
"""
complete_log = []

if Path("complete_log.json").exists():
    with open("complete_log.json", "r") as f:
        complete_log = json.load(f)

print("episodes", len(episodes))
for episode in episodes:
    print("episode:", episode)

    # uuid of episode
    json_file = list(episode.glob("*json"))[0]
    with open(json_file, "r") as f:
        metadata = json.load(f)
    uuid = metadata["uuid"]

    # extract date
    regex = r'\w+\+\w+\+(\d+-\d+-\d+-\w+-\w+-\w+)$'
    date_str = re.findall(regex, uuid)[0]
    date = dt.datetime.strptime(date_str, "%Y-%m-%d-%Hh-%Mm-%Ss")

    # organisation
    org = uuid.split("+")[0]

    plot_path = Path("plot") / (uuid + ".jpg")
    print("plot path:", plot_path)

    command = ["src/raw.py", "--visualize", "--scene", str(episode), "--plot", plot_path] # INTER
    print(f'Running: "{" ".join(map(str, command))}"')
    p: subprocess.CompletedProcess = subprocess.run(command)

    # read json of completed episode
    with open("single_log.json", "r") as f:
        single_episode = json.load(f)

    complete_log.append({
        "uuid": uuid,
        "path": str(episode),
        "org": org,
        "day": f"{date.year}-{date.month:0>2}-{date.day:0>2}",
        "instruction": annotations[uuid]["language_instruction1"],
        "visible_points": single_episode["visible_points"],
        "gripper_closed_times": single_episode["gripper_closed_times"],
        "gripper_duration": single_episode["gripper_duration"],
        "episode_duration": single_episode["episode_duration"]
    })

    with open("complete_log.json", "w") as f:
        json.dump(complete_log, f, indent=4, ensure_ascii=False)

    input("continue") # INTER