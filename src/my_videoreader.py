#!/usr/bin/env python3

# Based on raw.py.
# Simplified class to only read video frames without logging to rerun.

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
import skimage
from scipy import ndimage

from .common import h5_tree, CAMERA_NAMES, log_angle_rot, blueprint_row_images, extract_extrinsics, log_cartesian_velocity, POS_DIM_NAMES, link_to_world_transform
from .rerun_loader_urdf import URDFLogger
from .my_image_saver import ImageSaver

class StereoCamera:
    left_images: list[np.ndarray]
    right_images: list[np.ndarray]
    depth_images: list[np.ndarray]
    width: float
    height: float
    left_dist_coeffs: np.ndarray
    left_intrinsic_mat: np.ndarray

    right_dist_coeffs: np.ndarray
    right_intrinsic_mat: np.ndarray

    def __init__(self, recordings: Path, serial: int):
        
        try:
            import pyzed.sl as sl
            init_params = sl.InitParameters()
            svo_path = recordings / "SVO" / f"{serial}.svo"
            init_params.set_from_svo_file(str(svo_path))
            init_params.depth_mode = sl.DEPTH_MODE.QUALITY
            init_params.svo_real_time_mode = False
            init_params.coordinate_units = sl.UNIT.METER
            init_params.depth_minimum_distance = 0.2

            zed = sl.Camera()
            err = zed.open(init_params)
            if err != sl.ERROR_CODE.SUCCESS:
                raise Exception(f"Error reading camera data: {err}")

            params = (
                zed.get_camera_information().camera_configuration.calibration_parameters
            )
            
            self.left_intrinsic_mat = np.array(
                [
                    [params.left_cam.fx, 0, params.left_cam.cx],
                    [0, params.left_cam.fy, params.left_cam.cy],
                    [0, 0, 1],
                ]
            )
            self.right_intrinsic_mat = np.array(
                [
                    [params.right_cam.fx, 0, params.right_cam.cx],
                    [0, params.right_cam.fy, params.right_cam.cy],
                    [0, 0, 1],
                ]
            )
            self.zed = zed
        except ModuleNotFoundError:
            # pyzed isn't installed we can't find its intrinsic parameters
            # so we will have to make a guess.
            self.left_intrinsic_mat = np.array([
                [733.37261963,   0.,         625.26251221],
                [  0.,         733.37261963,  361.92279053],
                [  0.,           0.,           1.,        ]
            ])
            self.right_intrinsic_mat = self.left_intrinsic_mat
            
            mp4_path = recordings / "MP4" / f'{serial}-stereo.mp4'
            if (recordings / "MP4" / f'{serial}-stereo.mp4').exists():
                mp4_path = recordings / "MP4" / f'{serial}-stereo.mp4'
            elif (recordings / "MP4" / f'{serial}.mp4').exists():
                # Sometimes they don't have the '-stereo' suffix
                mp4_path = recordings / "MP4" / f'{serial}.mp4'
            else:
                raise Exception(f"unable to video file for camera {serial}")

            self.cap = cv2.VideoCapture(str(mp4_path))
            # print(f"opening {mp4_path}")


    def get_next_frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gets the the next from both cameras and maybe computes the depth."""

        if hasattr(self, "zed"):
            # We have the ZED SDK installed.
            import pyzed.sl as sl
            left_image = sl.Mat()
            right_image = sl.Mat()
            depth_image = sl.Mat()

            rt_param = sl.RuntimeParameters()
            err = self.zed.grab(rt_param)
            if err == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(left_image, sl.VIEW.LEFT)
                left_image = np.array(left_image.numpy())

                self.zed.retrieve_image(right_image, sl.VIEW.RIGHT)
                right_image = np.array(right_image.numpy())

                self.zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)
                depth_image = np.array(depth_image.numpy())
                return (left_image, right_image, depth_image)
            else:
                return None
        else:
            # We don't have the ZED sdk installed
            ret, frame = self.cap.read()
            if ret:
                left_image = frame[:,:1280,:]
                right_image = frame[:,1280:,:]
                return (left_image, right_image, None)
            else:
                # print("empty!")
                return None


class RawScene:
    dir_path: Path
    trajectory_length: int
    metadata: dict
    cameras: dict[str, StereoCamera]

    def __init__(self,
                 dir_path: Path,
                 visualize: bool
            ):
        self.dir_path = dir_path
        # MV
        self.visualize = visualize

        json_file_paths = glob.glob(str(self.dir_path) + "/*.json")
        if len(json_file_paths) < 1:
            raise Exception(f"Unable to find metadata file at '{self.dir_path}'")

        with open(json_file_paths[0], "r") as metadata_file:
            self.metadata = json.load(metadata_file)

        self.trajectory = h5py.File(str(self.dir_path / "trajectory.h5"), "r")
        self.action = self.trajectory['action']

        # We ignore the robot_state under action/, don't know why where is two different robot_states.
        self.robot_state = self.trajectory['observation']['robot_state']
        if self.visualize:
            h5_tree(self.trajectory)

        self.trajectory_length = self.metadata["trajectory_length"]

        # Mapping from camera name to it's serial number.
        self.serial = {
            camera_name: self.metadata[f"{camera_name}_cam_serial"]
            for camera_name in CAMERA_NAMES
        }

        self.cameras = {}
        for camera_name in CAMERA_NAMES:
            self.cameras[camera_name] = StereoCamera(
                self.dir_path / "recordings",
                self.serial[camera_name]
            )
        
        # MV compute FPS
        time_stamp_list = self.trajectory["observation"]["timestamp"][
            "cameras"
        ][f"{self.serial[camera_name]}_estimated_capture"]
        frame_count = len(time_stamp_list)
        t_total = float(time_stamp_list[-1] - time_stamp_list[0]) / 1000 # microsec -> sec
        print("episode sec", int(t_total))
        print("frames", frame_count)
        print("fps {:.2f}".format(frame_count / t_total)) # usually 14

        # MV
        # draw trajectory on 2D image
        self.points = []  # [(y, x, color)]
        self.first_touch = -1  # step number
        self.finger_tip: np.array = np.eye(4)  # [4, 4] link_to_world

        # gripper statistics
        self.FPS = int(14.15)
        self.is_gripper_closed = False
        self.gripper_close_count = 0
        self.gripper_duration = 0
        self.visible_count = 0  # is projected 2D point visible

        # compute difference image
        self.first_touch_3d: np.array = None
        self.first_touch_2d: np.array = None
        self.max_distance_grip: float = 0

        self.imsaver: ImageSaver = ImageSaver()

    def log_cameras_next(self, i: int) -> None:
        """
        Log data from cameras at step `i`.
        It should be noted that it logs the next camera frames that haven't been 
        read yet, this means that this method must only be called once for each step 
        and it must be called in order (log_cameras_next(0), log_cameras_next(1)). 

        The motivation behind this is to avoid storing all the frames in a `list` because
        that would take up too much memory.
        """

        return_dict = {}

        for camera_name, camera in self.cameras.items():
            # MV
            if camera_name != "ext1":
               continue

            # MV compute gripper state
            l = len(self.action['gripper_position'])
            signal = self.action['gripper_position'][max(0, i-int(self.FPS*0.8)):min(l, i+int(self.FPS*0.8))]
            gripper_on = np.sum(signal > 0.5)
            if gripper_on == len(signal):
                # gripper always on
                if self.is_gripper_closed == False:
                    self.gripper_close_count += 1

                if self.first_touch == -1:
                    self.first_touch = i

                self.is_gripper_closed = True
            elif gripper_on == 0:
                # gripper always off
                self.is_gripper_closed = False

            if self.is_gripper_closed:
                self.gripper_duration += 1
            # END

            time_stamp_camera = self.trajectory["observation"]["timestamp"][
                "cameras"
            ][f"{self.serial[camera_name]}_estimated_capture"][i]
            # rr.set_time_nanos("real_time", time_stamp_camera * int(1e6))

            frames = camera.get_next_frame()
            if frames:
                left_image, right_image, depth_image = frames

                # MV
                left_image = left_image[:, :, ::-1]
                imginfo = lambda img: print(type(img), img.dtype, img.shape, img.min(), img.max())

                # Ignore points that are far away.

                return_dict[f"cameras/{camera_name}/left"] = left_image
        
        return return_dict


def main():
    # MV
    parser = argparse.ArgumentParser(
        description="Visualizes the DROID dataset using Rerun."
    )

    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--plot", default="data/plot.jpg", type=Path)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument("--urdf", default="franka_description/panda.urdf", type=Path)
    args = parser.parse_args()

    raw_scene: RawScene = RawScene(args.scene, args.visualize)
    images: dict = raw_scene.log_grip_frame()

    plot_path = "data/tmp.jpg"
    cv2.imwrite(plot_path, cv2.cvtColor(images["cameras/ext1/left"], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])
    input("continue")

if __name__ == "__main__":
    main()
