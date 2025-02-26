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
    print("annotations before cleaning:", len(annotations.keys()))
    
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
    print("existing annotations:", len(annotations.keys()))


    # === filter large date groups
    def uuid_to_date(uuid):
        # IPRL+w026bb9b+2023-04-20-23h-28m-09s
        regex = r'\w+\+\w+\+(\d+-\d+-\d+-\w+-\w+-\w+)$'
        date_str = re.findall(regex, uuid)[0]

        # organisation
        org = uuid.split("+")[0]

        import datetime as dt
        return {
            "date": dt.datetime.strptime(date_str, "%Y-%m-%d-%Hh-%Mm-%Ss"),
            "org": org.lower()
        }


    def date_filter(date_uuid_path):
        _episode_date = []
        _episode_org = []
        for _date, uuid, _path in date_uuid_path:
            # extract date from uuid
            _info = uuid_to_date(uuid)
            _episode_date.append(_info["date"])
            _episode_org.append(_info["org"])

        filtered = []
        w = 200
        for i in range(len(date_uuid_path)):
            close_cnt = 0
            _left = max(0, i - w)
            _right = _left + 2 * w
            if _right >= len(date_uuid_path):
                _right = len(date_uuid_path)
                _left = _right - 2 * w
            for j in range(_left, _right):
                diff_sec = (_episode_date[i] - _episode_date[j]).total_seconds()
                if diff_sec < 3600 * 24 and _episode_org[i] == _episode_org[j]:
                    close_cnt += 1
            if close_cnt > w // 2:
                filtered.append(date_uuid_path[i])
        return filtered

    #existing_groupped = date_filter(existing_episodes)
    existing_groupped = existing_episodes

    # === filter by annotation
    # ordered by increasing date
    selected_episodes = {} # {"IPRL+w026bb9b+2023-04-20-23h-28m-09s": {"language_instruction1": ...}}
    no_annotation_cnt = 0
    for _date, uuid, _path in existing_groupped:
        matches = False
        
        if uuid not in annotations:
            no_annotation_cnt += 1
            continue

        for annot_key in annotations[uuid]:
            annot = annotations[uuid][annot_key].lower() # very important!
            regex1 = r"(take|remove|from).*(cup|mug|pot|bowl)"
            regex2 = r"move.*(forward|backwards|left|right)"
            regex3 = r"(close|drawer|blocks|charger|adapter)"
            regex4 = r"(rope|cable|towel|cloth|rubber band)"
            regex5 = r"(door|spoon|kettle|curtain|hang|pillow|fold|push|press|tissue|scoop|cook|stir|switch)"
            if (
                len(annot) > 200
                # or re.findall(regex1, annot)
                # or re.findall(regex2, annot)
                # or re.findall(regex3, annot)
                or re.findall(regex4, annot)
                or re.findall(regex5, annot)
            ):
                matches = False
                break

            if (
                #"marker" in annot
                #"autolab" in uuid.lower()
                "block" in annot
                #re.findall(regex5, annot)
            ):
                save_key = annot_key
                matches = True
        if matches:
            selected_episodes[uuid] = annotations[uuid][save_key]
    print("episodes without annotation:", no_annotation_cnt)
    print("selected:", len(selected_episodes))
    selected_list = list(selected_episodes.keys())
    selected_list = selected_list[:10000]

    #selected_annotations = {uuid : annotations[uuid] for uuid in selected_list}
    selected_annotations = [[i, uuid, annotations[uuid]] for i, uuid in enumerate(selected_list)]
    with open("data/selected_annotations.json", "w") as f:
        json.dump(selected_annotations, f, indent=4, ensure_ascii=False)
    print("to download:", len(selected_list))
    input("continue...")

    # === download
    for iteration, uuid in enumerate(selected_list):
        # IPRL+w026bb9b+2023-04-20-23h-28m-09s
        print("=== download {:,d} of {:,d}, uuid".format(iteration, len(selected_list)),  uuid)

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
