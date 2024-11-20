from pathlib import Path
import json
import argparse
import re
import datetime
import numpy as np

import skimage
from skimage import io
import PIL
from torchvision.transforms import v2

from .raw import RawScene, scene_to_date
from .my_sam import DetectionResult, DetectionProcessor, plot_detections
from .my_episode_list import manual_paths, date_to_localpath, manual_dates, train_idx, val_idx

def draw_sequence(image: np.ndarray, points: list):
    """
    :param points: [(x, y, color)]
        color presets: 0 - begin green, end red;
        1, 2 - fixed colors.
    """
    colors = {}
    colors[0] = np.array([255, 0, 0])
    colors[1] = np.array([0, 255, 0])
    colors[2] = np.array([0, 0, 255])

    canvas = np.copy(image)
    for i, (x, y, color) in enumerate(points):
        rows, cols = skimage.draw.disk((y, x), 8, shape=canvas.shape)
        k = i / len(points)

        if color == 0:
            # mix
            canvas[rows, cols, :3] = colors[1] * (1-k) + colors[0] * k
        else:
            # fixed color
            canvas[rows, cols, :3] = colors[color]

    return canvas



class DrawTrajectory:
    def __init__(self, dir_path: str):
        self.raw_scene: RawScene = RawScene(dir_path, False)
        _images: dict = self.raw_scene.log_cameras_next(0)
        self.image = _images["cameras/ext1/left"]

        # dig out intrinsics from ZED
        _camera = self.raw_scene.cameras["ext1"]
        _intrinsics: np.ndarray = _camera.left_intrinsic_mat # (3, 3)
        _pinhole = np.eye(4)[:3, :4]
        self.ip = _intrinsics @ _pinhole # (3, 4)
    
    def set_trajectory(self, trajectory: np.ndarray):
        trajectory = trajectory[:, :4]
        trajectory[:, 3] = 1
        self.trajectory = trajectory # (n_steps, 4)

    def draw(self):
        points = []
        for i in range(len(self.trajectory)):
            _point_3d = self.trajectory[i] # [X, Y, Z, 1]
            _point_2d = self.ip @ _point_3d # (3, 4) x (4, 1) = (3, 1)
            _point_2d = _point_2d / _point_2d[2] # [x/z, y/z, 1]
            _x, _y = _point_2d[0], _point_2d[1]
            points.append((_x, _y, 0))
        self.image = draw_sequence(self.image, points)

    def save(self, image_path: str):
         io.imsave(image_path, self.image, quality=90)



def plot_2d_evaluation():
    trajectory_dir = Path("data/eval/trajectory")
    train_dir = Path("data/eval/train")
    val_dir = Path("data/eval/val")
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    print("plot train")
    for i in train_idx:
        _episode_date: str = manual_dates[i]
        _path = trajectory_dir / (_episode_date + ".npy")
        if _path.exists():
            print("train ", i)
            with open(_path, "rb") as f:
                trajectory = np.load(f)
            _scene = date_to_localpath[_episode_date]
            plotter = DrawTrajectory(_scene)
            plotter.set_trajectory(trajectory)
            plotter.draw()
            plotter.save(train_dir / (_episode_date + ".jpg"))
    print("plot val")
    for i in val_idx:
        _episode_date: str = manual_dates[i]
        _path = trajectory_dir / (_episode_date + ".npy")
        if _path.exists():
            print("val ", i)
            with open(_path, "rb") as f:
                trajectory = np.load(f)
            _scene = date_to_localpath[_episode_date]
            plotter = DrawTrajectory(_scene)
            plotter.set_trajectory(trajectory)
            plotter.draw()
            plotter.save(val_dir / (_episode_date + ".jpg"))

def rerun_evaluation():
    selected_sid = [0]
    for sid in selected_sid:
        _episode_date = manual_dates[sid]
        _path = Path("data/eval/trajectory") / (_episode_date + ".npy")
        with open(_path, "rb") as f:
            trajectory = np.load(f)
        _ = RawScene(manual_paths[sid], True, trajectory)


if __name__ == "__main__":
    plot_2d_evaluation()
    rerun_evaluation()
