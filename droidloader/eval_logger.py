# Save trajectory during evaluation.

from pathlib import Path
import json
import argparse
import re
import datetime
import numpy as np

from .my_episode_list import manual_dates



class EvalResults:
    def __init__(self):
        episode_idx: int = None
        my_data: np.ndarray = None
        obs: np.ndarray = None
        pred: np.ndarray = None
eval_results = EvalResults()



class EvalLogger:
    def __init__(self, save_dir="data/eval/trajectory"):
        self.save_dir = save_dir
        self.episode_date: str
        self.trajectory: list[np.ndarray] = []

    def vis_start(self, episode_idx: int):
        self.episode_date = manual_dates[episode_idx]
        self.trajectory = []

    def vis_step(self, prediction: np.ndarray):
        # collect whole trajectory in list

        self.trajectory.append(prediction)

    def vis_stop(self):
        # save collected trajectory to npy

        self.trajectory = np.stack(self.trajectory)

        _trajectory_path = Path(self.save_dir + "/" + self.episode_date + ".npy")
        _trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        with open(_trajectory_path, "wb") as f:
            np.save(f, self.trajectory)

