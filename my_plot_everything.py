from pathlib import Path
import json
import re
import datetime as dt
import subprocess
import shutil
import argparse

def main():
    # ====== PREPARATIONAL PART ======

    parser = argparse.ArgumentParser(
        "Plot trajectory for all downloaded episodes"
    )
    parser.add_argument('--visualize', action='store_true', help="show rerun visualisation")
    args = parser.parse_args()

    root_dir = Path(__file__).parent
    target_dir = root_dir / "data" / "droid_raw" / "1.0.1"
    annotations_file_name = "aggregated-annotations-030724.json"
    with open(target_dir / annotations_file_name) as f:
        annotations = json.load(f)
    
    # [ date "2023-03-02-15h-14m-31s", uuid "IRIS+ef107c48+2023-03-02-15h-14m-31s", path IRIS/success/(date)/(time) ]
    # with open("data/existing_episodes.json") as f:
    #     existing_episodes = json.load(f)

    # list of available episodes
    data = Path("data/droid_raw/1.0.1/success")
    episodes = []
    for date in sorted(data.iterdir()):
        for episode in sorted(date.iterdir()):
            # data/droid_raw/1.0.1/success/2023-03-02/Thu_Mar__2_15_00_02_2023
            # .    .         .     .       date       episode
            episodes.append(episode)
            print(f"{len(episodes)-1: >4}", episode)
    episodes = episodes[:1]

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



    for episode in episodes:
        # ====== PREPARATIONAL PART ======

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




        # ====== ACTUAL PROCESSING ======

        plot_path = Path("plot") / (date_str + ".jpg")
        print("plot path:", plot_path)

        # === SAM
        from src.process_imitation_flow import process_trajectory
        process_trajectory(episode)

        # === trajectory plot
        command = ["python", "-m", "src.raw", "--scene", str(episode), "--plot", plot_path]
        if args.visualize:
            command.append("--visualize")
        print(f'Running: "{" ".join(map(str, command))}"')
        p: subprocess.CompletedProcess = subprocess.run(command)

        # === first frame
        # data_dir = Path("data")
        # # file_list = sorted((data_dir / "frames").glob("center*jpg"))
        # plot_path = Path("plot/f1") / (date_str + ".jpg")
        # plot_path.parent.mkdir(parents=True, exist_ok=True)
        # shutil.copy2("data/frames/first_image.jpg", plot_path)



        # ====== SAVE ======

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

        if args.visualize:
            input("continue")

if __name__ == "__main__":
    main()
