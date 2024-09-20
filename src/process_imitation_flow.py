# Copied from imitation_flow_nick.ipynb

import sys
from pathlib import Path
from typing import List, Dict

import ipywidgets
import numpy as np
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt

import casino
#from DITTO.data import Hands23Dataset, get_all_runs
#from DITTO.config import BASE_RECORDING_PATH, TIME_STEPS
# from DITTO.tracking_3D import Step3DMethod



from DITTO.trajectory import Trajectory

from .my_loader import DroidLoader
from pathlib import Path

def process_trajectory(scene):
    # src/raw.py --visualize  --scene data/droid_raw/1.0.1/success/2023-04-07/Fri_Apr__7_13_32_40_2023
    # scene =                          "data/droid_raw/1.0.1/success/2023-03-08/Wed_Mar__8_16_45_10_2023"
    loader = DroidLoader(Path(scene))
    loader.read_trajectory()
    loaders: List = [loader]



    num_frames = -1 # TIME_STEPS  # number of frames through which we compute flow
    trajectories: Dict[int, Trajectory] = {}
    for demonstration_index in tqdm(range(len(loaders))):
        trajectories[demonstration_index] = Trajectory.from_hands23(loaders[demonstration_index], n_frames=num_frames)

    # We could pre compute trajectories with .trajectory_2D and .trajectory_3D
    trajectory = trajectories[0].trajectory_2D
    # we need n points, not n - 1
    start, stop = loader.get_start_stop()
    n = stop - start + 1
    full_trajectory = np.zeros((n, 1, 2))
    full_trajectory[1:n] = trajectory
    full_trajectory[0] = trajectory[0]
    trajectory = full_trajectory



    with open("data/trajectory.npy", "wb") as f:
        np.save(f, trajectory)

