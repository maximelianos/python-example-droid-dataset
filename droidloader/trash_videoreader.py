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

from my_videoreader import RawScene
from my_sam import DetectionResult, DetectionProcessor, plot_detections

import PIL

class DroidLoader:
    def __init__(self, scene: str):
        self.i: int = 0
        self.image: np.array = {}
        self.detection: DetectionResult = None
        self.is_gripper_closed = False

        self.rgb = []
        self.start = -1
        self.stop = -1

        # === extract frames
        self.raw_scene: RawScene = RawScene(scene, False)
        images: dict = raw_scene.log_cameras_next(0)
        self.i += 1
        self.image = images["cameras/ext1/left"]

        # debug
        imginfo = lambda img: print(type(img), img.dtype, img.shape, img.min(), img.max())
        # imginfo(images["cameras/ext1/left"])
        # plot_path = "data/tmp.jpg"
        # cv2.imwrite(plot_path, cv2.cvtColor(images["cameras/ext1/left"], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])

        # === detect objects
        detector_id = "IDEA-Research/grounding-dino-base"
        segmenter_id = "facebook/sam-vit-base"
        processor = DetectionProcessor(detector_id, segmenter_id)

        labels = ["a pen."]
        threshold = 0.3

        # image = images["cameras/ext1/left"] # cv2.imread(image_url)[:,:,::-1].astype(np.float32) / 255
        # MV debug
        # plot_path = Path("data") / "frame.jpg"
        # cv2.imwrite(plot_path, cv2.cvtColor(image * 255, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])

        image_array, detections = processor.grounded_segmentation(
            image=PIL.Image.fromarray(self.image),
            labels=labels,
            threshold=threshold,
            polygon_refinement=True,
        )

        plot_detections(image_array, detections, "data/segmentation.jpg")

        # save detection in this class member variable
        if detections:
            self.detection = detections[0]
        else:
            self.detection = None

    def read_trajectory(self):
        # === read all frames into memory...

        for i in range(1, self.raw_scene.trajectory_length):
            # ban gripper closing for the second time
            if self.raw_scene.gripper_close_count > 1:
                break

            # limit trajectory length
            if len(self.rgb) > 10:
                break

            # read frame
            images: dict = self.raw_scene.log_cameras_next(i)

            # skip when not closed
            if not self.raw_scene.is_gripper_closed:
                continue

            if self.start == -1:
                self.start = i
            self.stop = i
            self.rgb.append(images["cameras/ext1/left"])

    def get_start_stop(self) -> tuple[int, int]:
        return (self.start, self.stop)

    def get_timesteps(self, n_frames: int) -> list[int]:
        return list(range(self.start, self.stop))

    def get_rgb(self, timestamp: int) -> np.ndarray:
        return self.rgb[timestamp - self.start]

    def get_depth(self, timestamp: int) -> np.ndarray:
        return np.zeros_like(self.rgb[0])

    def get_object_mask(self, timestamp: int) -> np.ndarray:
        # return uint8 (h, w) [0, 255]
        return self.detection.mask

    def get_bbox(self, demo_start: int, object_key: str):
        box = self.detection.box
        return [box.xmin, box.xmax, box.ymin, box.ymax]


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
    print(loader.get_start_stop())
    print(loader.get_timesteps())
    imginfo(loader.get_rgb())
    imginfo(loader.get_depth())
    imginfo(loader.get_object_mask())
    print(loader.get_bbox(start, "hand_bbox"))


    # TODO
    # loader.get_start_stop() -> [int, int]
    # loader.get_timesteps(n_frames: int) -> list[int]
    # loader.get_rgb(timestamp) -> np.array
    # loader.get_depth(timestamp) -> np.array
    # loader.get_object_mask(timestamp) -> np.array
    # loader.get_goal_mask(int) -> np.array
    # loader.get_bbox(demo_start: int, "hand_bbox") -> [x_start, x_stop, y_start, y_stop]


if __name__ == "__main__":
    main()