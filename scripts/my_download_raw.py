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

    # MV: created special file to convert uuid to path
    # === result scheme
    # === { "IRIS+ef107c48+2023-03-02-15h-14m-31s": IRIS/success/(date)/(time) }
    with open("existing_episodes.json") as f:
        existing_episodes = json.load(f)

    print("episodes:", len(annotations.keys()))
    selected_episodes = {} # {"IPRL+w026bb9b+2023-04-20-23h-28m-09s": {"instruction1":..., "instruction2:..., "instruction3":...}}
    for date in annotations:
        for lang_i in annotations[date]:
            description = annotations[date][lang_i]
            if "put" in description and "marker" in description and len(description) < 50:
                selected_episodes[date] = annotations[date]
                break

    print("selected:", len(selected_episodes))
    selected_list = list(selected_episodes.keys())[::10][:100] # each 10-th episode, at most 100

    for key in selected_list:
        # IPRL+w026bb9b+2023-04-20-23h-28m-09s

        if not key in existing_episodes:
            continue

        regex = r'\w+\+\w+\+(\d+-\d+-\d+-\w+-\w+-\w+)$'
        date_str = re.findall(regex, key)[0]

        import datetime as dt
        date = dt.datetime.strptime(date_str, "%Y-%m-%d-%Hh-%Mm-%Ss")

        print("=== download", key)
        #date = datetime.strptime(date, "%Y-%b-%d_%H:%M:%S")
        formated_date = date.strftime("%Y-%m-%d-%Hh-%Mm-%Ss")

        org = None
        for key_a in annotations.keys():
            if formated_date in key_a:
                org = key_a.split("+")[0]

        rel_path = f"success/{date.year}-{date.month:0>2}-{date.day:0>2}"

        # BAD CODE BEGIN
        def day_to_str(day: int):
            return f"{day:_>2}"

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
        # for src_path in variants:

        # BAD CODE END

        # === gs://gresearch/robotics/droid_raw/1.0.1/ <- root for existing_episodes
        src_path = f"gs://gresearch/robotics/droid_raw/1.0.1/" + existing_episodes[key]
        dst_path = target_dir / rel_path
        dst_path.mkdir(parents=True, exist_ok=True)
        command = ["gsutil", "-m", "cp", "-n", "-r", src_path, dst_path]
        print(f'Running: "{" ".join(map(str, command))}"')
        input("continue")
        p: subprocess.CompletedProcess = subprocess.run(command)

        if p.returncode == 0:
            print("success!")

        input("continue")

if __name__ == "__main__":
    main()
