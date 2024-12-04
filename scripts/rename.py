from pathlib import Path
import shutil

data_dir = Path("data/projection")
paths = sorted(data_dir.iterdir())
for i, src in enumerate(paths):
    dst = data_dir / f"{i:03d}.jpg"
    shutil.move(src, dst)


