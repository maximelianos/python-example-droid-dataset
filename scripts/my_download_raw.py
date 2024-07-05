#!/usr/bin/env python3

from pathlib import Path
import subprocess
import argparse
import json
from datetime import datetime
import re

def launch_cmd(cmds, stdout=False, show=False) -> subprocess.CompletedProcess:
    """
    :param cmds: list of commands
    :param stdout: bool, return stdout as str
    :param show: bool, write to stdout
    :return: None
    """
    proc: subprocess.CompletedProcess
    for cmd in cmds:
        print(cmd)
        if stdout:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        elif show:
            proc = subprocess.run(cmd)
        else:
            proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if proc.returncode != 0:
            return proc
    return proc

def main():
    parser = argparse.ArgumentParser(
        "Downloads a succesfull scene from the raw version of the dataset"
    )
    parser.add_argument("--out", default=None, type=Path, help="where to store data, by default it gets puts in data/")
    parser.add_argument(
        "--date", type=str, help="date of recording, format: %Y_%b_%d_%H:%M:%S"
    )



    args = parser.parse_args()

    if args.out is None:
        root_dir = Path(__file__).parent.parent
        target_dir = root_dir / "data" / "droid_raw" / "1.0.1"
    else:
        target_dir = args.out

    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)

    annotations_file_name = "aggregated-annotations-030724.json"
    if not (target_dir / annotations_file_name).exists():
        command = [
            "gsutil",
            "-m",
            "cp",
            f"gs://gresearch/robotics/droid_raw/1.0.1/{annotations_file_name}",
            str(target_dir),
        ]
        print(f'Annotation file not found, running {" ".join(command)}')
        subprocess.run(command)

    with open(target_dir / annotations_file_name) as f:
        annotations = json.load(f)
    
    print("episodes:", len(annotations.keys()))
    selected_episodes = {}
    for date in annotations:
        for lang_i in annotations[date]:
            description = annotations[date][lang_i]
            if "put" in description and "marker" in description and len(description) < 50:
                selected_episodes[date] = annotations[date]
                break

    print("selected:", len(selected_episodes))

    # each 10-th episode, at most 100
    to_select = list(selected_episodes.keys())[::10][:100]
    selected_s = {}
    for date in to_select:
        selected_s[date] = selected_episodes[date]
    #print(json.dumps(selected_s, indent=4))

    for key in to_select:
        # IPRL+w026bb9b+2023-04-20-23h-28m-09s

        regex = r'\w+\+\w+\+(\d+-\d+-\d+-\w+-\w+-\w+)$'
        date_str = re.findall(regex, key)[0]

        import datetime as dt
        date = dt.datetime.strptime(date_str, "%Y-%m-%d-%Hh-%Mm-%Ss")

        # download selected episode
        print("=== download", key)
        #date = datetime.strptime(date, "%Y-%b-%d_%H:%M:%S")
        formated_date = date.strftime("%Y-%m-%d-%Hh-%Mm-%Ss")


        org = None
        for key in annotations.keys():
            if formated_date in key:
                org = key.split("+")[0]

        def day_to_str(day: int):
            return f"{day:_>2}"


        rel_path = f"success/{date.year}-{date.month:0>2}-{date.day:0>2}"

        variants = [
            f"gs://gresearch/robotics/droid_raw/1.0.1/{org}/"
            + rel_path
            + "/"
            + date.strftime("%a_%b_%d_%H:%M:%S_%Y"),

            f"gs://gresearch/robotics/droid_raw/1.0.1/{org}/"
            + rel_path
            + "/"
            + date.strftime("%a_%b_%d_%H:%M:%S_%Y"),

            f"gs://gresearch/robotics/droid_raw/1.0.1/{org}/"
            + rel_path
            + "/"
            # + date.strftime("%a_%b_%d_%H:%M:%S_%Y")
            # Thu_Mar__2_14_57_27_2023
            + date.strftime("%a_%b_")
            + day_to_str(date.day)
            + date.strftime("_%H_%M_%S_%Y"),
        ]

        for src_path in variants:
            dst_path = target_dir / rel_path
            dst_path.mkdir(parents=True, exist_ok=True)
            command = ["gsutil", "-m", "cp", "-n", "-r", src_path, dst_path]
            print(f'Running: "{" ".join(map(str, command))}"')
            p: subprocess.CompletedProcess = subprocess.run(command)

            if p.returncode == 0:
                print("sucess!")
                break

        #input("continue")

if __name__ == "__main__":
    main()
