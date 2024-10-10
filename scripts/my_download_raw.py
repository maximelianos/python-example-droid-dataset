#!/usr/bin/env python3
# Select episodes based on text annotation and download.
# Based on download_raw.py

from pathlib import Path
import subprocess
import argparse
import json
from datetime import datetime
import re

def main():
    parser = argparse.ArgumentParser(
        "Downloads a succesfull scene from the raw version of the dataset"
    )
    parser.add_argument("--out", default=None, type=Path, help="where to store data, by default in data/")
    parser.add_argument('--debug', action='store_true', help="stop on points")
    args = parser.parse_args()
    if args.out is None:
        # ./scripts/my_download_raw.py
        root_dir = Path(__file__).parent.parent
        target_dir = root_dir / "data" / "droid_raw" / "1.0.1"
    else:
        target_dir = args.out
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
        # scheme { str uuid: {"language_instruction1": str, ...} }
        annotations = json.load(f)
    print("episodes before cleaning:", len(annotations.keys()))

    # === existing episodes [date, uuid, path]
    with open("data/existing_episodes.json") as f:
        existing_episodes = json.load(f)
    uuid_to_path = {episode[1]: episode[2] for episode in existing_episodes}

    # remove non-existing episodes
    intersection = {}
    for uuid in annotations:
        if uuid in uuid_to_path:
            intersection[uuid] = annotations[uuid]
    annotations = intersection
    print("episodes after cleaning:", len(annotations.keys()))



    # === filter based on annotation
    # ordered by increasing date
    selected_episodes = {} # {"IPRL+w026bb9b+2023-04-20-23h-28m-09s": {"language_instruction1": ...}}
    no_annotation_cnt = 0
    for _date, uuid, _path in existing_episodes:
        is_good = True
        to_save = False
        
        if uuid not in annotations:
            no_annotation_cnt += 1
            continue

        for annot_key in annotations[uuid]:
            annot = annotations[uuid][annot_key].lower() # very important!
            regex1 = r"(take|remove|from).*(cup|mug|pot|bowl)"
            regex2 = r"move.*(forward|backwards|left|right)"
            regex3 = r"(close|drawer)"
            if len(annot) > 60 or (
                #re.findall(regex1, annot) or re.findall(regex2, annot)
                re.findall(regex3, annot)
            ):
               is_good = False
            
            is_match = (
                #"marker" in annot
                "block" in annot
            )
            if is_match:
                save_key = annot_key
                to_save = True
        if is_good and to_save:
            selected_episodes[uuid] = annotations[uuid][save_key]
    print("no annotations:", no_annotation_cnt)
    print("selected:", len(selected_episodes))
    selected_list = list(selected_episodes.keys())
    #selected_list = selected_list[455:456] # 810:1000

    # for i, uuid in enumerate(selected_list):
    #     print(i, uuid_to_ind[uuid])

    #selected_annotations = {uuid : annotations[uuid] for uuid in selected_list}
    selected_annotations = [[i, uuid, annotations[uuid]] for i, uuid in enumerate(selected_list)]
    with open("data/selected_annotations.json", "w") as f:
        json.dump(selected_annotations, f, indent=4, ensure_ascii=False)
    print("to download:", len(selected_list))
    input("continue...")

    # === download
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

        # === gs://gresearch/robotics/droid_raw/1.0.1/ <- root
        src_path = f"gs://gresearch/robotics/droid_raw/1.0.1/" + uuid_to_path[uuid]
        dst_path = target_dir / rel_path
        dst_path.mkdir(parents=True, exist_ok=True)
        command = ["gsutil", "-m", "cp", "-n", "-r", src_path, dst_path]
        print(f'Running: "{" ".join(map(str, command))}"')
        p: subprocess.CompletedProcess = subprocess.run(command)

        if p.returncode == 0:
            print("success!")

        if args.debug:
            input("continue")

if __name__ == "__main__":
    main()
