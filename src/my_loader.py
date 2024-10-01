# Detect and segment first frame,
# return frames during grip

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

from .raw import RawScene
from .my_sam import DetectionResult, DetectionProcessor, plot_detections



imginfo = lambda img: print(type(img), img.dtype, img.shape, img.min(), img.max())

def scene_to_date(scene: str):
    # uuid of episode
    json_file = list(Path(episode).glob("*json"))[0]
    with open(json_file, "r") as f:
        metadata = json.load(f)
    uuid = metadata["uuid"]

    # extract date
    regex = r'\w+\+\w+\+(\d+-\d+-\d+-\w+-\w+-\w+)$'
    date_str = re.findall(regex, uuid)[0]
    date = dt.datetime.strptime(date_str, "%Y-%m-%d-%Hh-%Mm-%Ss")

    # organisation
    org = uuid.split("+")[0]

    return date

class DroidLoader:
    def __init__(self, scene: str):
        self.i: int = 0
        self.image: np.array = {}
        self.detection: DetectionResult = DetectionResult()
        self.is_gripper_closed = False

        self.rgb = []
        self.start = -1
        self.stop = -1

        self.intrinsics = None

        # === extract frames
        self.raw_scene: RawScene = RawScene(scene, False)
        images: dict = self.raw_scene.log_cameras_next(0)
        self.i += 1
        self.image = images["cameras/ext1/left"]

        # === detect objects
        # check if detection was already performed
        episode_date: str = scene_to_date(scene)
        mask_path = Path("data/detection/" + episode_date + "_mask.npy")
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        if mask_path.exists():
            with open(mask_path, "rb") as f:
                self.detection.mask = np.load(f)

            return

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

        # save detection in this class member variable
        if detections:
            self.detection = detections[0]
        else:
            self.detection.mask = np.zeros(())
        
        # cache mask, load cache in same function
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

            if self.start == -1:
                self.start = 0
            self.stop = len(self.rgb) - 1

            self.rgb.append(images["cameras/ext1/left"])

    def get_start_stop(self) -> tuple[int, int]:
        return (0, self.stop - self.start)

    def get_timesteps(self, n_frames: int) -> list[int]:
        return list(range(0, self.stop - self.start))

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

        # check if trajectory was already computed
        episode_date: str = scene_to_date(scene)
        trajectory_path = Path("data/trajectory/" + episode_date + "_traj.npy")
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        if trajectory_path.exists():
            with open(mask_path, "rb") as f:
                self.trajectory = np.load(f)

            return self.trajectory

        # src/raw.py --visualize  --scene data/droid_raw/1.0.1/success/2023-04-07/Fri_Apr__7_13_32_40_2023
        # scene =                          "data/droid_raw/1.0.1/success/2023-03-08/Wed_Mar__8_16_45_10_2023"
        Path("data/trajectory.npy").unlink(missing_ok=True)
        self.read_trajectory()
        loaders: List = [self]


        num_frames = -1 # TIME_STEPS  # number of frames through which we compute flow
        trajectories: Dict[int, Trajectory] = {}
        for demonstration_index in tqdm(range(len(loaders))):
            trajectories[demonstration_index] = Trajectory.from_hands23(loaders[demonstration_index], n_frames=num_frames)

        # We could pre compute trajectories with .trajectory_2D and .trajectory_3D
        trajectory = trajectories[0].trajectory_2D
        # we need n points, not n - 1
        start, stop = self.get_start_stop()
        n = stop - start + 1
        full_trajectory = np.zeros((n, 1, 2))
        full_trajectory[1:n] = trajectory
        full_trajectory[0] = trajectory[0]
        trajectory = full_trajectory


        with open("data/trajectory.npy", "wb") as f:
            np.save(f, trajectory)

        with open(trajectory_path, "wb") as f:
            np.save(f, trajectory)
        
        self.trajectory = trajectory
        return trajectory


def main():
    # === parse arguments
    parser = argparse.ArgumentParser(
        description="Visualizes the DROID dataset using Rerun."
    )

    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    loader = DroidLoader(args.scene)
    loader.read_trajectory()
    start, stop = loader.get_start_stop()
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
