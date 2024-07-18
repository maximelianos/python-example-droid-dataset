from pathlib import Path
import json
import re
import datetime as dt
import subprocess
import shutil

from difference import Difference


def main():
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
    episodes = episodes[3:]

    exclude = [
        "2023-04-27/Thu_Apr_27_22:35:22_2023", # failed download?
        "2023-04-27/Thu_Apr_27_22:41:17_2023",
        "2023-04-27/Thu_Apr_27_23:07:13_2023",
        "2023-04-27/Thu_Apr_27_23:13:21_2023"
    ]
    for ex_ep in exclude:
        for i in range(len(episodes)):
            if ex_ep in str(episodes[i]):
                episodes.pop(i)
                print("removed", ex_ep)
                break


    print("episodes:", len(episodes))
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

    if Path("data/complete_log.json").exists():
        with open("data/complete_log.json", "r") as f:
            complete_log = json.load(f)

    # MV
    # diff_processor = Difference()

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

        # difference
        # diff_processor.process()
        # Path("plotdiff").mkdir(parents=True, exist_ok=True)
        # plot_path = Path("plotdiff") / (uuid + ".jpg")
        # shutil.copy2("data/frames/result_overlay.jpg", plot_path)

        # read json of completed episode
        with open("data/single_log.json", "r") as f:
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

        with open("data/complete_log.json", "w") as f:
            json.dump(complete_log, f, indent=4, ensure_ascii=False)

        input("continue") # INTER

if __name__ == "__main__":
    main()