from pathlib import Path
import argparse
import json

def collect_yellow_block(directory: str, json_path: str):
    """
    Collect names of images from folder and save into json.
    Name format: 2023-10-09-11h-05m-13s_first.jpg
    """
    episodes = [path.stem.split("_")[0] for path in sorted(Path(directory).glob("*_first.jpg"))]
    with open(json_path, "w") as f:
        json.dump(episodes, f, indent=4, ensure_ascii=False)

parser = argparse.ArgumentParser(
    description="Collect list of wanted episodes from image filenames into json."
)
parser.add_argument("--dir", required=True, type=Path, help="directory with images")
args = parser.parse_args()

json_path = Path("data/manual_yellow_block.json")
collect_yellow_block(args.dir, json_path)

