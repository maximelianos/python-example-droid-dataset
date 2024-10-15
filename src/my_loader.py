# Detect and segment first frame,
# return frames during grip.
# Pytorch dataloader class.

import numpy as np
from pathlib import Path
import rerun as rr
import cv2
from scipy.spatial.transform import Rotation
from skimage import io
import glob
import h5py
import json
import argparse
import PIL
import re
import datetime as dt

import torch

from .raw import RawScene, scene_to_date
from .my_sam import DetectionResult, DetectionProcessor, plot_detections

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



imginfo = lambda img: print(type(img), img.dtype, img.shape, img.min(), img.max())

class DroidLoader:
    def __init__(self, scene: Path):
        # cache of mask in data/detection/<date>_mask.npy,
        # cache of box in data/detection/<date>_box.json,
        # cache of trajectory in data/trajectory/<date>_traj.npy

        self.scene = scene
        self.i: int = 0
        self.image: np.ndarray = None
        self.detection: DetectionResult = DetectionResult(None, None, None)
        self.is_gripper_closed = False

        self.rgb = []
        self.start = 0
        self.stop = -1

        self.intrinsics = None

        # === read frames
        self.raw_scene: RawScene = RawScene(scene, False)
        images: dict = self.raw_scene.log_cameras_next(0)
        self.i += 1
        self.image = images["cameras/ext1/left"]

        # === detect objects
        # check if detection was already performed
        episode_date: str = scene_to_date(scene)
        mask_path = Path("data/detection/" + episode_date + "_mask.npy")
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        box_path = Path("data/detection/" + episode_date + "_box.json")
        if box_path.exists():
            # load box
            with open(box_path, "r") as f:
                self.detection = DetectionResult.from_dict(json.load(f))
            
            # load mask
            with open(mask_path, "rb") as f:
                self.detection.mask = np.load(f)

            return

        # === run detection only if no cache
        detector_id = "IDEA-Research/grounding-dino-base"
        segmenter_id = "facebook/sam-vit-base"
        processor = DetectionProcessor(detector_id, segmenter_id)

        labels = ["a green cube."]
        threshold = 0.3

        # === debug
        # image = images["cameras/ext1/left"] # cv2.imread(image_url)[:,:,::-1].astype(np.float32) / 255
        # plot_path = Path("data") / "frame.jpg"
        # cv2.imwrite(plot_path, cv2.cvtColor(image * 255, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])

        # detections: zero or one detection List[DetectionResult]
        # DetectionResult: mask
        image_array, detections = processor.grounded_segmentation(
            image=PIL.Image.fromarray(self.image),
            labels=labels,
            threshold=threshold,
            polygon_refinement=True,
        )

        plot_path = Path("data/segmentation.jpg")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_detections(image_array, detections, str(plot_path))

        # === save detection in this class member variable
        if detections:
            self.detection = detections[0]
        else:
            self.detection.mask = np.zeros(())
        
        # cache box
        with open(box_path, "w") as f:
            json.dump(self.detection.to_dict(), f, indent=4, ensure_ascii=False)
        
        # cache mask
        with open(mask_path, "wb") as f:
            np.save(f, self.detection.mask)

    def read_trajectory(self):
        # === read all frames into memory...

        for i in range(1, self.raw_scene.trajectory_length):
            # limit trajectory length
            if len(self.rgb) >= 70:
                break

            # read frame
            images: dict = self.raw_scene.log_cameras_next(i)

            # ban gripper closing for the second time
            if self.raw_scene.gripper_close_count > 1:
                break

            # skip when not closed
            if not self.raw_scene.is_gripper_closed:
                continue

            # skip frames in between
            #if (len(self.rgb) + i) % 4 != 0:
            #    continue

            self.stop = len(self.rgb)

            self.rgb.append(images["cameras/ext1/left"])

    def get_start_stop(self) -> tuple[int, int]:
        # last index not included
        return (0, self.stop)  

    def get_timesteps(self, n_frames: int) -> list[int]:
        return list(range(0, self.stop))

    def get_rgb(self, timestamp: int) -> np.ndarray:
        return self.rgb[timestamp]

    def get_depth(self, timestamp: int) -> np.ndarray:
        return np.zeros_like(self.rgb[0])

    def get_object_mask(self, timestamp: int, refined=False) -> np.ndarray:
        # return uint8 (h, w, 1) [0, 255]
        return self.detection.mask[:, :, np.newaxis]

    def get_goal_mask(self, timestamp: int, refined=False) -> np.ndarray:
        # return uint8 (h, w, 1) [0, 255]
        return self.detection.mask[:, :, np.newaxis]

    def get_bbox(self, demo_start: int, object_key: str):
        box = self.detection.box
        return [box.xmin, box.xmax, box.ymin, box.ymax]
    


    def track(self) -> np.ndarray:
        # return trajectory [n, 1, 2]
        # Copied from imitation_flow_nick.ipynb

        # === check if trajectory was already computed
        episode_date: str = scene_to_date(self.scene)
        trajectory_path = Path("data/trajectory/" + episode_date + "_traj.npy")
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        if trajectory_path.exists():
            with open(trajectory_path, "rb") as f:
                self.trajectory = np.load(f)

            return self.trajectory

        # === compute trajectory
        loaders: List = [self]
        num_frames = -1 # TIME_STEPS  # number of frames through which we compute flow
        trajectories: Dict[int, Trajectory] = {}
        for demonstration_index in tqdm(range(len(loaders))):
            trajectories[demonstration_index] = Trajectory.from_hands23(loaders[demonstration_index], n_frames=num_frames)

        # We could pre compute trajectories with .trajectory_2D and .trajectory_3D
        trajectory = trajectories[0].trajectory_2D
        print("trajectory shape", end=" ")
        imginfo(Trajectory)
        input("debug now!")


        # we need n points, not n - 1
        
        # start, stop = self.get_start_stop() 
        # n = stop - start
        # full_trajectory = np.zeros((n, 1, 2))
        # full_trajectory[1:n] = trajectory
        # full_trajectory[0] = trajectory[0]
        # trajectory = full_trajectory

        with open(trajectory_path, "wb") as f:
            np.save(f, trajectory) 
        self.trajectory = trajectory

        return trajectory


class EpisodeList(torch.utils.data.Dataset):
    def __init__(self):
        # === read list of espisodes which was saved by dirlist.py
        from .my_episode_list import date_to_localpath

        with open("data/manual_episodes.json", "r") as f:
            date_list = json.load(f)
            self.path_list = [date_to_localpath[date] for date in date_list]

    def __getitem__(self, idx: int):
        loader = DroidLoader(Path(self.path_list[idx]))
        loader.read_trajectory()
        loader.track()
        images = [torch.from_numpy(image) for image in loader.rgb] # list[(h, w, c)]
        images = [tensor.permute(2, 0, 1) for tensor in images] # list[(c, h, w)]
        images = torch.stack(images) # (n, c, h, w)
        sample = {
            "images": images,
            "robot_state": loader.trajectory[:, 0, :] # (n, 2)
        }
        return sample





def main():
    # === Test DroidLoader
    parser = argparse.ArgumentParser(
        description="Visualizes the DROID dataset using Rerun."
    )

    parser.add_argument("--scene", required=False, type=Path)
    args = parser.parse_args()

    eplist = EpisodeList()
    cur_scene: Path
    if args.scene:
        cur_scene = args.scene
    else:
        cur_scene = Path(eplist.path_list[0])

    loader = DroidLoader(cur_scene)
    loader.read_trajectory()
    start, _ = loader.get_start_stop()
    print("start, stop", loader.get_start_stop())
    print("timesteps", loader.get_timesteps(0))
    print("rgb", end=" ")
    imginfo(loader.get_rgb(start))
    print("depth", end=" ")
    imginfo(loader.get_depth(start))
    print("mask", end=" ")
    imginfo(loader.get_object_mask(start))
    print("bbox", loader.get_bbox(start, "hand_bbox"))
    print("trajectory", end=" ")
    imginfo(loader.track())

    # === Test EpisodeList
    sample = eplist[0]
    print("sample")
    imginfo(sample["images"])
    imginfo(sample["robot_state"])

    # Interface
    # loader.get_start_stop() -> [int, int]
    # loader.get_timesteps(n_frames: int) -> list[int]
    # loader.get_rgb(timestamp) -> np.array
    # loader.get_depth(timestamp) -> np.array
    # loader.get_object_mask(timestamp) -> np.array (h, w, 1)
    # loader.get_goal_mask(int) -> np.array
    # loader.get_bbox(demo_start: int, "hand_bbox") -> [x_start, x_stop, y_start, y_stop]


if __name__ == "__main__":
    main()
