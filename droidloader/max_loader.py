# Max dataloader

import json
from typing import is_typeddict
import numpy as np
from pathlib import Path

from skimage import io
from .my_episode_list import imginfo, manual_dates

def load_npy(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return np.load(f)

imginfo = lambda img: print(type(img), img.dtype, img.shape, img.min(), img.max())

class MaxLoader:
    def __init__(self):
        pass

    def __len__(self):
        return len(manual_dates)
    
    def get_info(self, idx) -> dict[str, list]:
        _path = Path("data/trajectory/" + manual_dates[idx] + ".json")
        with open(_path, "r") as f:
            info = json.load(f)
        info["camera_intrinsic"] = np.array(info["camera_intrinsic"])
        info["camera_extrinsic"] = np.array(info["camera_extrinsic"])
        info["obj_start_pose"] = np.array(info["obj_start_pose"])
        info["obj_end_pose"] = np.array(info["obj_end_pose"])
        return info

    def get_start_stop_point(self, idx) -> tuple[np.ndarray, np.ndarray]:
        # return (array(3), array(3))
        _path = Path("data/trajectory/" + manual_dates[idx] + "_traj3d.npy")
        t = load_npy(_path) # (n_steps, 4)
        t = t[:3]  # remove color
        return (t[0], t[-1])
    
    def get_image_first(self, idx) -> np.ndarray:
        # return array(h, w, 3)
        _path = Path("data/trajectory/" + manual_dates[idx] + "_first.jpg")
        return io.imread(_path).astype(np.float32) / 255

    def get_image_grip(self, idx) -> np.ndarray:
        # return array(h, w, 3)
        _path = Path("data/trajectory/" + manual_dates[idx] + "_grip.jpg")
        return io.imread(_path).astype(np.float32) / 255


def test_loader():
    loader = MaxLoader(True)
    print("number of episodes:", len(loader))
    print("=== Shuffle episode #0")
    print("annotation:", loader.get_annotation(0))
    print("info:")
    print(loader.get_info(0))
    #print("start and stop point:")
    #print(loader.get_start_stop_point(0))
    print("first image:")
    imginfo(loader.get_image_0(0))

if __name__ == "__main__":
    test_loader()
