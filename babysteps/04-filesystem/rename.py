from pathlib import Path
import shutil

data_dir = Path("source")
target_dir = Path("target")
for i, filename in sorted(data_dir.iterdir()):
    target = target_dir / f"{i:03d}.jpg"
    shutil.move(filename, target)
