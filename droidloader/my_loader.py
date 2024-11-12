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
from torchvision.transforms import v2

from .raw import RawScene, scene_to_date
from .my_sam import DetectionResult, DetectionProcessor, plot_detections
from .my_episode_list import manual_paths

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
    def __init__(self, scene: str):
        # cache of mask in data/detection/<date>_mask.npy,
        # cache of box in data/detection/<date>_box.json,
        # cache of trajectory in data/trajectory/<date>_traj.npy

        self.scene = scene
        self.image: np.ndarray = None
        self.detection: DetectionResult = DetectionResult(None, None, None)
        self.is_gripper_closed = False

        self.rgb = []
        self.depth = []
        self.full_pcd = []
        self.pcd = []
        self.start = 0
        self.stop = -1

        self.intrinsics = None

        # === read first frame
        self.raw_scene: RawScene = RawScene(scene, False)
        images: dict = self.raw_scene.log_cameras_next(0)
        self.image = images["cameras/ext1/left"]

        # dig out intrinsic from ZED
        _camera = self.raw_scene.cameras["ext1"]
        _left_intrinsic: np.ndarray = _camera.left_intrinsic_mat
        self.intrinsics = casino.pointcloud.Intrinsics.from_matrix(_left_intrinsic)

        # === check if detection was already performed
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
        self.detection = detections[0]
        
        # cache box
        with open(box_path, "w") as f:
            json.dump(self.detection.to_dict(), f, indent=4, ensure_ascii=False)
        
        # cache mask
        with open(mask_path, "wb") as f:
            np.save(f, self.detection.mask)

    def _gripper_frames(self):
        # Return only frames where gripper is closed.
        # returns one frame or None
        self.raw_scene = RawScene(self.scene, False)   # reset the reader

        for i in range(0, self.raw_scene.trajectory_length):
            # limit trajectory length
            if len(self.rgb) >= 70:
                return

            # read frame
            images: dict = self.raw_scene.log_cameras_next(i)

            # ban gripper closing for the second time
            if self.raw_scene.gripper_close_count > 1:
                return

            # skip when not closed
            if not self.raw_scene.is_gripper_closed:
                continue

            # skip frames in between
            #if (len(self.rgb) + i) % 4 != 0:
            #    continue

            yield images

    def read_trajectory(self):
        # read all frames into memory...
        self.rgb = []
        self.depth = []
        self.full_pcd = []
        self.pcd = []

        for images in self._gripper_frames():
            self.rgb.append(images["cameras/ext1/left"])
            self.depth.append(images["cameras/ext1/depth"])
            self.full_pcd.append(images["cameras/ext1/full_pcd"])
            self.pcd.append(images["cameras/ext1/pcd"])

        self.stop = len(self.rgb)

    def get_start_stop(self) -> tuple[int, int]:
        # last index not included
        return (0, self.stop)  

    def get_timesteps(self, n_frames: int) -> list[int]:
        return list(range(0, self.stop))

    def get_rgb(self, timestamp: int) -> np.ndarray:
        return self.rgb[timestamp]

    def get_depth(self, timestamp: int) -> np.ndarray:
        return self.depth[timestamp]

    def get_object_mask(self, timestamp: int, refined=False) -> np.ndarray:
        # return uint8 (h, w, 1) [0, 255]
        return self.detection.mask[:, :, np.newaxis] > 128

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
        self.episode_date: str = scene_to_date(self.scene)
        trajectory_path = Path("data/trajectory/" + self.episode_date + "_traj.npy")
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
        trajectory = trajectories[0].trajectory_2D  # (n_steps, 1, 2)

        trajectory = trajectory.reshape((-1, 2))

        with open(trajectory_path, "wb") as f:
            np.save(f, trajectory)
        with open("data/trajectory.npy", "wb") as f:
            np.save(f, trajectory) 
        self.trajectory = trajectory

        return trajectory

    def track_3d(self):
        # Read 2D trajectory, use point cloud, and save 3D trajectory
        trajectory = self.trajectory
        n_steps, _ = trajectory.shape

        # === check if trajectory was already computed
        traj3d_path = Path("data/trajectory/" + self.episode_date + "_traj3d.npy")
        if traj3d_path.exists():
            with open(traj3d_path, "rb") as f:
                self.trajectory_3d = np.load(f)
            return self.trajectory_3d

        # calculate 3D trajectory
        self.trajectory_3d = np.zeros((n_steps, 4)) # list[numpy(XYZ+RGBA)]
        for i, images in enumerate(self._gripper_frames()):
            y, x = trajectory[i]
            _pcd = images["cameras/ext1/full_pcd"]
            self.trajectory_3d[i] = _pcd[y, x]
        # fill nans
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
        self.trajectory_3d = nan_filler(self.trajectory_3d)

        with open(traj3d_path, "wb") as f:
            np.save(f, self.trajectory_3d)
        with open("data/trajectory_3d.npy", "wb") as f:
            np.save(f, self.trajectory_3d)
        return self.trajectory_3d

    def save_pcd(self):
        # call .read_trajectory before this!
        # otherwise point cloud will be uncut
        pcd_path = Path("data/trajectory/" + self.episode_date + "_pcd.npy")
        _p = [point_cloud[:, :3] for point_cloud in self.pcd] # cut out color
        _p: np.ndarray = np.stack(_p) # (n_steps, n_points, XYZ)
        with open(pcd_path, "wb") as f:
            np.save(f, _p)


 

class EpisodeList:
    # DO NOT USE, SLOW!

    def __init__(self):
        # === read list of espisodes which was saved by dirlist.py
        self.path_list = manual_paths

    def __getitem__(self, idx: int):
        loader = DroidLoader(self.path_list[idx])
        loader.read_trajectory()
        loader.track()

        # all images in episode torch
        images = [torch.from_numpy(image) for image in loader.rgb] # list[torch (h, w, c)]
        h, w, c = images[0].shape
        print("image")
        imginfo(images[0])
        images = [image.permute(2, 0, 1) for image in images] # list[torch (c, h, w)]
        images = [v2.Resize(size=(128, 128))(image).permute(1, 2, 0) for image in images] # backbone requirement (128, 128)
        images = torch.stack(images).float() / 255 # (n, h, w, c)

        # === trajectory
        _t = loader.track_3d() # (n_steps, 4)
        MAX_STEPS = 20
        _t = _t[:MAX_STEPS, :3] # remove color
        pad_width = ((0, 0), (0, 7)) # pad robot state
        _t = np.pad(_t, pad_width, mode="constant")
        pad_width = ((0, MAX_STEPS-len(_t)), (0, 0)) # pad n_steps
        trajectory = np.pad(_t, pad_width, mode="edge")


        # === pcd numpy
        # list[(n_points, XYZ+color)] -> list[(n_points, XYZ)]
        # n_points must be same for all pcds
        _p = [point_cloud[:, :3] for point_cloud in loader.pcd]
        _p: np.ndarray = np.stack(_p) # (n_steps, n_points, XYZ)

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
        _p = _p[:MAX_STEPS] # limit to n_steps
        pad_width = ((0, MAX_STEPS-len(_p)), (0, 0), (0, 0))
        pcds = np.pad(_p, pad_width, mode="edge")

        sample = {
            "images": images,
            "pcd_xyz": pcds, # (n_steps, n_points, XYZ)
            "robot_state": trajectory # (n, 2)
        }
        return sample


def process_manuals():
    for scene in manual_paths[:2]:
        print("=== PROCESSING SCENE", scene)
        Path("data/trajectory.npy").unlink(missing_ok=True)
        Path("data/trajectory_3d.npy").unlink(missing_ok=True)
        loader = DroidLoader(scene)
        loader.read_trajectory()
        loader.track()
        loader.track_3d() # [4] = XYZ+color
        loader.read_trajectory() # raw.py will cut pcd now
        loader.save_pcd()




def main():
    # === Test DroidLoader
    parser = argparse.ArgumentParser(
        description="Visualizes the DROID dataset using Rerun."
    )

    parser.add_argument("--scene", required=False, type=Path)
    parser.add_argument("--sid", required=False, type=int)
    args = parser.parse_args()

    scene: str
    if args.sid is not None:
        scene = manual_paths[args.sid]
    else:
        scene = args.scene

    loader = DroidLoader(scene)
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
    print("mask")
    imginfo(loader.detection.mask)
    print("trajectory 3d", end=" ")
    imginfo(loader.track_3d())
    print(loader.track_3d())

    print("intrinsics", end=" ")
    print(loader.intrinsics.matrix)


    with open("data/trajectory.npy", "wb") as f:
        np.save(f, loader.trajectory)
    with open("data/trajectory_3d.npy", "wb") as f:
        np.save(f, loader.trajectory_3d)


if __name__ == "__main__":
    #main()
    process_manuals()
