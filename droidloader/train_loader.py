from typing import is_typeddict
import numpy as np
from pathlib import Path
from .my_episode_list import manual_dates

def load_npy(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return np.load(f)

imginfo = lambda img: print(type(img), img.dtype, img.shape, img.min(), img.max())

class EpisodeList:
    def __init__(self, is_train):
        # === read list of espisodes which was saved by dirlist.py
        self.is_train = is_train
        self.is_test = True
        if self.is_test:
            self.ind_list = range(0, 10)
        else:
            if is_train:
                self.ind_list = range(0, 130)
            else:
                self.ind_list = range(130, 139)
        self.date_list = [manual_dates[i] for i in self.ind_list]

        self.MAX_STEPS = 34

    def __len__(self):
        if self.is_test:
            return len(self.date_list)
        else:
            return len(self.date_list) * (self.MAX_STEPS - 1)

    def __getitem__(self, idx: int):
        # Duplicate each episode by starting from different time.
        # Number of items is len(date_list) * MAX_STEPS
        print("sample idx", idx)
        
        if self.is_test:
            _episode_idx = idx
            _start_step = 0
        else:
            _episode_idx = idx // self.MAX_STEPS
            _start_step = idx % self.MAX_STEPS
            #print(_episode_idx, _start_step)

        self.manual_idx = self.ind_list[_episode_idx]
        self.episode_date = self.date_list[_episode_idx]

        # === trajectory
        path = Path("data/trajectory/" + self.episode_date + "_traj3d.npy")
        _t = load_npy(path) # (n_steps, 4)
        _start = min(_start_step, len(_t) - 1) # don't cut the beginning too much
        _t = _t[_start:_start+self.MAX_STEPS, :3] # start from different step, total length < MAX_STEPS, remove color
        pad = np.array([0.407, -0.652, .639, # all points have same rotation. Took data from Eugenio
            -0.909,  -0.354,  -0.218, 1.0])
        pad_width = ((0, 0), (0, 7)) # pad robot state
        _t = np.pad(_t, pad_width, mode="constant") # (n_steps, 10)
        _t[:, 3:] = pad.reshape((1, 7))
        pad_width = ((0, self.MAX_STEPS-len(_t)), (0, 0)) # pad n_steps
        trajectory = np.pad(_t, pad_width, mode="edge")


        # === pcd numpy
        # list[(n_points, XYZ+color)] -> list[(n_points, XYZ)]
        # n_points must be same for all pcds
        path = Path("data/trajectory/" + self.episode_date + "_pcd.npy")
        _p = load_npy(path) # (n_steps, n_point, XYZ)

        def nan_helper(y):
            """Helper to handle indices and logical indices of NaNs.

            Input:
                - y, 1d numpy array with possible NaNs
            Output:
                - nans, logical indices of NaNs
                - index, a function, with signature indices= index(logical_indices),
                to convert logical indices of NaNs to 'equivalent' indices
                Example:
                >>> # linear interpolation of NaNs
                >>> nans, x= nan_helper(y)
                >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
            """

            return np.isnan(y), lambda z: z.nonzero()[0]

        def nan_filler(y: np.ndarray) -> np.ndarray:
            """Fill nans along 1st dimension.

            y: (n_steps, ...)
            """
            shape = y.shape
            y = y.reshape((shape[0], -1)).transpose()  # (emb, n_steps)
            nans, x = nan_helper(y)
            y[nans] = np.interp(x(nans), x(~nans), y[~nans])
            y = y.transpose().reshape(shape) # original shape
            return y

        _p = nan_filler(_p) # fill nans along n_steps dimension
        _p = _p[_start:_start+self.MAX_STEPS] # limit to n_steps
        pad_width = ((0, self.MAX_STEPS-len(_p)), (0, 0), (0, 0))
        pcds = np.pad(_p, pad_width, mode="edge")

        if self.is_test:
            from .eval_logger import eval_results
            eval_results.episode_idx = self.manual_idx
            eval_results.my_data = trajectory

        sample = {
            "pcd_xyz": pcds, # (n_steps, n_points, XYZ)
            "robot_state": trajectory # (n, 3+7)
        }
        return sample

def main():
    # === Test EpisodeList
    eplist = EpisodeList(is_train=True)
    for i in range(len(eplist)):
        sample = eplist[i]
        #print("robot state batch", end=" ")
        #imginfo(sample["robot_state"])
        #print("pcd")
        #imginfo(sample["pcd_xyz"])

if __name__ == "__main__":
    main()
