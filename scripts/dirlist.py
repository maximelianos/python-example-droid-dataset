# Collect list of wanted episodes from images directory into json.

from pathlib import Path
import argparse
import json

parser = argparse.ArgumentParser(
    description="Collect list of wanted episodes from images directory into json."
)
parser.add_argument("--dir", required=True, type=Path, help="directory with images")
args = parser.parse_args()

episodes = [path.stem for path in sorted(Path(args.dir).glob("*py"))]
json_path = Path("data/manual_episodes.json")
with open(json_path, "w") as f:
    json.dump(episodes, f, indent=4, ensure_ascii=False)
