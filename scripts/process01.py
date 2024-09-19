# Create list of existing episodes sorted by date
# [
#   [ date "2023-03-02-15h-14m-31s", uuid "IRIS+ef107c48+2023-03-02-15h-14m-31s", path IRIS/success/(date)/(time) ]
# ]
# Produce data/existing_episodes.json
# Prerequisite: download text .json files (300 mb)

from pathlib import Path
import argparse
import json
import re

parser = argparse.ArgumentParser(
    "Convert downloaded json descriptions to one json file"
)
parser.add_argument("--data", default=None, type=Path, help="path to droid dataset")

args = parser.parse_args()

# === gs://gresearch/robotics/droid_raw/1.0.1/AUTOLab/success/2023-12-19/Tue_Dec_19_10:41:57_2023/
# === gs://gresearch/robotics/droid_raw/1.0.1/ <- root
droid_root = Path(args.data)
result = []
for org in droid_root.iterdir():
    if not org.is_dir():
        # unwanted files
        continue

    success_path = org / "success"

    for date in sorted(success_path.iterdir()):

        for time in sorted(date.iterdir()):
            # handle missing metadata
            json_files = list(time.glob("*json"))
            if len(json_files) == 0:
                continue

            json_file = json_files[0]
            with open(json_file, "r") as f:
                metadata = json.load(f)

            if not metadata["success"]:
                continue

            # extract date from uuid
            regex = r'\w+\+\w+\+(\d+-\d+-\d+-\w+h-\w+m-\w+s)$'
            date_str = re.findall(regex, metadata["uuid"])[0]
            result.append([
                date_str, metadata["uuid"], str(time.relative_to(droid_root))
            ])

# sort by date
result = sorted(result)

Path("data").mkdir(parents=True, exist_ok=True)
with open("data/existing_episodes.json", "w") as f:
    json.dump(result, f, indent=4, ensure_ascii=False)