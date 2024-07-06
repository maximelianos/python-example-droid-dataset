# download only text .json files (300 mb)
# $ gsutil -m rsync -r -x "(.*mp4)|(.*svo)|(.*h5)|(failure)|(timestamp)" gs://gresearch/robotics/droid_raw/1.0.1/  droid_raw
# $ python scripts/process01.py --data ../droid_raw/

from pathlib import Path
import argparse
import json

parser = argparse.ArgumentParser(
    "Convert downloaded json descriptions to one json file"
)
parser.add_argument("--data", default=None, type=Path, help="path to droid dataset")

args = parser.parse_args()

# === gs://gresearch/robotics/droid_raw/1.0.1/AUTOLab/success/2023-12-19/Tue_Dec_19_10:41:57_2023/
# === gs://gresearch/robotics/droid_raw/1.0.1/ <- root
droid_root = Path(args.data)


# === result scheme
# === { "IRIS+ef107c48+2023-03-02-15h-14m-31s": IRIS/success/(date)/(time) }
result = {}

for org in droid_root.iterdir():
    if not org.is_dir():
        continue

    success_path = org / "success"

    for date in sorted(success_path.iterdir()):

        for time in sorted(date.iterdir()):
            json_files = list(time.glob("*json"))
            if len(json_files) == 0:
                continue

            json_file = json_files[0]
            metadata: object
            with open(json_file, "r") as f:
                metadata = json.load(f)

            if not metadata["success"]:
                continue

            result[metadata["uuid"]] = str(time.relative_to(droid_root))

with open("existing_episodes.json", "w") as f:
    json.dump(result, f, indent=4, ensure_ascii=False)