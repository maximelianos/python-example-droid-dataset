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
    print("episodes before cleaning:", len(annotations.keys()))

    # MV: created special file to convert uuid to path
    # === result scheme
    # === { "IRIS+ef107c48+2023-03-02-15h-14m-31s": IRIS/success/(date)/(time) }
    with open("data/existing_episodes.json") as f:
        existing_episodes = json.load(f)

    # remove all non-existing episodes!
    intersection = {}
    for uuid in annotations:
        if uuid in existing_episodes:
            intersection[uuid] = annotations[uuid]
    annotations = intersection
    print("episodes after cleaning:", len(annotations.keys()))

    # === filter based on annotation
    selected_episodes = {} # {"IPRL+w026bb9b+2023-04-20-23h-28m-09s": {"language_instruction1": ...}}
    for date in annotations:
        is_good = True
        to_save = False
        
        for annot_key in annotations[date]:
            annot = annotations[date][annot_key].lower() # very important!
            regex1 = r"(take|remove|from).*(cup|mug|pot|bowl)"
            regex2 = r"move.*(forward|backwards|left|right)"
            if len(annot) > 60 or re.findall(regex1, annot) or re.findall(regex2, annot):
                is_good = False
            
            is_match = (
                "marker" in annot
            )
            if is_match:
                save_key = annot_key
                to_save = True
        if is_good and to_save:
            selected_episodes[date] = annotations[date][save_key]
    print("selected:", len(selected_episodes))

    selected_list = sorted(list(selected_episodes.keys()))
    selected_list = selected_list[::len(selected_list) // 100][:100] # select 200 episodes uniformly

    selected_annotations = {uuid : annotations[uuid] for uuid in selected_list}
    with open("data/selected_annotations.json", "w") as f:
        json.dump(selected_annotations, f, indent=4, ensure_ascii=False)

    print("to download:", len(selected_list))
    input("continue...")

    for uuid in selected_list:
        # IPRL+w026bb9b+2023-04-20-23h-28m-09s
        print("=== download", uuid)

        # extract date
        regex = r'\w+\+\w+\+(\d+-\d+-\d+-\w+-\w+-\w+)$'
        date_str = re.findall(regex, uuid)[0]

        import datetime as dt
        date = dt.datetime.strptime(date_str, "%Y-%m-%d-%Hh-%Mm-%Ss")

        # organisation
        org = uuid.split("+")[0]

        # year-month-day
        rel_path = f"success/{date.year}-{date.month:0>2}-{date.day:0>2}"

        # === gs://gresearch/robotics/droid_raw/1.0.1/ <- root for existing_episodes
        src_path = f"gs://gresearch/robotics/droid_raw/1.0.1/" + existing_episodes[uuid]
        dst_path = target_dir / rel_path
        dst_path.mkdir(parents=True, exist_ok=True)
        command = ["gsutil", "-m", "cp", "-n", "-r", src_path, dst_path]
        print(f'Running: "{" ".join(map(str, command))}"')
        p: subprocess.CompletedProcess = subprocess.run(command)

        if p.returncode == 0:
            print("success!")

        input("continue")

if __name__ == "__main__":
    main()
