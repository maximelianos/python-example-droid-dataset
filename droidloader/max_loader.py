# Max dataloader

from typing import is_typeddict
import numpy as np
from pathlib import Path

from skimage import io
from .my_episode_list import date_to_uuid, annotations, imginfo
from .my_episode_list import manual_dates, train_idx, val_idx

def load_npy(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return np.load(f)

imginfo = lambda img: print(type(img), img.dtype, img.shape, img.min(), img.max())

class MaxLoader:
    def __init__(self, is_train):
        # === read list of espisodes which was saved by dirlist.py
        self.is_train = is_train
        self.is_test = False  # enable after training
        if self.is_test:
            if self.is_train:
                self.ind_list = []
            else:
                self.ind_list = np.concatenate((train_idx, val_idx))
        else:
            if is_train:
                self.ind_list = [0, 1, 2, 3, 4]
            else:
                self.ind_list = val_idx

    def __len__(self):
        return len(self.ind_list)
    
    def _shuffle_idx_to_date(self, idx):
        episode_idx = self.ind_list[idx]
        return  manual_dates[episode_idx]
    
    def get_annotation(self, idx) -> str:
        # return episode text annotation
        episode_date = self._shuffle_idx_to_date(idx)

        # scheme { str uuid: {"language_instruction1": str, ...} }
        _uuid: str = date_to_uuid[episode_date]
        return annotations[_uuid]["language_instruction1"]

    def get_intrinsic(self, idx) -> np.ndarray:
        # return np.ndarray (3, 3)
        episode_date = self._shuffle_idx_to_date(idx)

        _path = Path("data/trajectory/" + episode_date + "_intrinsic.npy")
        return load_npy(_path)

    def get_start_stop_point(self, idx) -> tuple[np.ndarray, np.ndarray]:
        # return (array(3), array(3))
        episode_date = self._shuffle_idx_to_date(idx)

        _path = Path("data/trajectory/" + episode_date + "_traj3d.npy")
        t = load_npy(_path) # (n_steps, 4)
        t = t[:3]  # remove color
        return (t[0], t[-1])
    
    
    def get_image_0(self, idx) -> np.ndarray:
        # return array(h, w, 3)
        episode_date = self._shuffle_idx_to_date(idx)

        _path = Path("data/trajectory/" + episode_date + "_img0.jpg")
        return io.imread(_path).astype(np.float32) / 255
    
        

def test_loader():
    loader = MaxLoader(True)
    print("number of episodes:", len(loader))
    print("=== Shuffle episode #0")
    print("annotation:", loader.get_annotation(0))
    print("intrinsic:")
    print(loader.get_intrinsic(0))
    print("start and stop point:")
    print(loader.get_start_stop_point(0))
    print("first image:")
    imginfo(loader.get_image_0(0))



if __name__ == "__main__":
    test_loader()
