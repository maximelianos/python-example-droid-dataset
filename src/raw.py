#!/usr/bin/env python3

# Read video and trajectory, pipe images to rerun.

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

from common import h5_tree, CAMERA_NAMES, log_angle_rot, blueprint_row_images, extract_extrinsics, log_cartesian_velocity, POS_DIM_NAMES, link_to_world_transform
from rerun_loader_urdf import URDFLogger
from my_image_saver import ImageSaver

import skimage
from scipy import ndimage

def ext_to_camera(t, rot):
    # matrix to transform world PoV into camera PoV, C^-1
    ext = np.eye(4)
    ext[0:3, 0:3] = rot.T
    ext[0:3, 3] = -rot.T @ t
    return ext

def ext_to_world(t, rot):
    # inverse matrix, C
    ext = np.eye(4)
    ext[0:3, 0:3] = rot
    ext[0:3, 3] = t
    return ext

def draw_sequence(image: np.array, points: list):
    """
    :param points: [(x, y, color)]
        color presets: 0 - begin green, end red;
        > 0 - fixed colors.
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
            canvas[rows, cols] = colors[1] * (1-k) + colors[0] * k
        else:
            # fixed color
            canvas[rows, cols] = colors[color]

    return canvas


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


    def get_next_frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
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

        # computed from nick
        self.calc_trajectory = None
        trajectory_path = "data/trajectory.npy"
        if Path(trajectory_path).exists():
            with open(trajectory_path, "rb") as f:
                self.calc_trajectory = np.load(f)  # (n, 1, 2) - y, x
                print("loaded the calculated trajectory")

    def log_cameras_next(self, i: int) -> None:
        """
        Log data from cameras at step `i`.
        It should be noted that it logs the next camera frames that haven't been 
        read yet, this means that this method must only be called once for each step 
        and it must be called in order (log_cameras_next(0), log_cameras_next(1)). 

        The motivation behind this is to avoid storing all the frames in a `list` because
        that would take up too much memory.
        """

        for camera_name, camera in self.cameras.items():
            # MV
            if camera_name != "ext1":
               continue

            # MV compute gripper state
            l = len(self.action['gripper_position'])
            signal = self.action['gripper_position'][
                     max(0, i - int(self.FPS * 0.8) + int(self.FPS * 0.5)):
                     min(self.trajectory_length, i + int(self.FPS * 0.8) + int(self.FPS * 0.5))
                     ]
            gripper_on = np.sum(signal > 0.5)
            if gripper_on == len(signal):
                # gripper on during interval
                if self.is_gripper_closed == False:
                    self.gripper_close_count += 1

                if self.first_touch == -1:
                    self.first_touch = i

                self.is_gripper_closed = True
            elif gripper_on == 0:
                # gripper off during interval
                self.is_gripper_closed = False


            if self.is_gripper_closed:
                self.gripper_duration += 1
            # END

            time_stamp_camera = self.trajectory["observation"]["timestamp"][
                "cameras"
            ][f"{self.serial[camera_name]}_estimated_capture"][i]
            rr.set_time_nanos("real_time", time_stamp_camera * int(1e6))

            # left view
            extrinsics_left = self.trajectory["observation"]["camera_extrinsics"][
                f"{self.serial[camera_name]}_left"
            ][i]
            rotation = Rotation.from_euler(
                "xyz", np.array(extrinsics_left[3:])
            ).as_matrix()

            rr.log(
                f"cameras/{camera_name}/left",
                rr.Pinhole(
                    image_from_camera=camera.left_intrinsic_mat,
                ),
            ),
            rr.log(
                f"cameras/{camera_name}/left",
                rr.Transform3D(
                    translation=np.array(extrinsics_left[:3]),
                    mat3x3=rotation,
                ),
            ),

            # MV compute projection matrix
            intr = camera.left_intrinsic_mat  # [3, 3]
            t = np.array(extrinsics_left[:3]) # [3]
            rot = rotation                    # [3, 3]

            pinhole = np.eye(4)[:3, :4]
            ext = ext_to_camera(t, rot)
            self.left_proj_mat = intr @ pinhole @ ext

            # === right view
            extrinsics_right = self.trajectory["observation"]["camera_extrinsics"][
                f"{self.serial[camera_name]}_right"
            ][i]
            rotation = Rotation.from_euler(
                "xyz", np.array(extrinsics_right[3:])
            ).as_matrix()

            rr.log(
                f"cameras/{camera_name}/right",
                rr.Pinhole(
                    image_from_camera=camera.right_intrinsic_mat,
                ),
            ),
            rr.log(
                f"cameras/{camera_name}/right",
                rr.Transform3D(
                    translation=np.array(extrinsics_right[:3]),
                    mat3x3=rotation,
                ),
            ),

            # === depth view
            depth_translation = (extrinsics_left[:3] + extrinsics_right[:3]) / 2
            rotation = Rotation.from_euler(
                "xyz", np.array(extrinsics_right[3:])
            ).as_matrix()

            rr.log(
                f"cameras/{camera_name}/depth",
                rr.Pinhole(
                    image_from_camera=camera.left_intrinsic_mat,
                ),
            ),
            rr.log(
                f"cameras/{camera_name}/depth",
                rr.Transform3D(
                    translation=depth_translation,
                    mat3x3=rotation,
                ),
            ),

            frames = camera.get_next_frame()
            if frames:
                left_image, right_image, depth_image = frames

                # MV
                left_image = left_image[:, :, ::-1]
                imginfo = lambda img: print(type(img), img.dtype, img.shape, img.min(), img.max())

                # save frame to queue
                self.imsaver.append(time_stamp_camera, left_image)  # time in ms
                self.imsaver.snap("first", left_image) # save first episode image

                # first projection
                point_3d = self.finger_tip @ [0, 0, 0, 1] # [4, 4] x [4], world coors
                point_3d = point_3d / point_3d[3]

                point_2d = self.left_proj_mat @ point_3d
                point_2d = point_2d / point_2d[2] # [x*z, y*z, z]
                x, y = point_2d[0], point_2d[1]
                

                if self.first_touch == i:
                    # frame of first touch
                    self.imsaver.snap("grip", left_image)
                    self.imsaver.save_center(time_stamp_camera)

                    # self.first_touch_3d = point_3d
                    self.first_touch_2d = point_2d

                # === after first touch
                if self.first_touch != -1:
                    self.points.append((x, y, 0))

                    if self.is_gripper_closed and self.gripper_close_count == 1:
                        cur_distance = np.sum((point_2d - self.first_touch_2d) ** 2)
                        if cur_distance > self.max_distance_grip:
                            self.max_distance_grip = cur_distance
                            self.imsaver.snap("max", left_image, replace=True)

                # === draw projection 1
                # left_image = draw_sequence(left_image, [(x, y, 2)])

                h, w, c = left_image.shape
                if 0 <= y < h and 0 <= x < w:
                    self.visible_count += 1

                # second projection
                cam = self.left_proj_mat @ (self.world_pos_3d @ [0, 0, 0, 1] ) # [x*z, y*z, z]
                cam = cam / cam[2]
                x, y = cam[0], cam[1]
                left_image = draw_sequence(left_image, [(x, y, 1)])

                # track point
                if (self.calc_trajectory and
                    self.first_touch != -1 and
                    i - self.first_touch < self.calc_trajectory.shape[0]
                ):
                    y, x = self.calc_trajectory[i - self.first_touch].reshape((2))
                    left_image = draw_sequence(left_image, [(x, y, 1)])

                # Ignore points that are far away.

                if self.visualize:
                    rr.log(f"cameras/{camera_name}/left", rr.Image(left_image))
                    # rr.log(f"cameras/{camera_name}/right", rr.Image(right_image))

                    if depth_image is not None:
                        depth_image[depth_image > 1.8] = 0
                        rr.log(f"cameras/{camera_name}/depth", rr.DepthImage(depth_image))

    def log_action(self, i: int) -> None:
        # MV
        pose = self.trajectory['observation']['robot_state']['cartesian_position'][i] # [6]
        # pose = self.trajectory['action']['robot_state']['cartesian_position'][i]
        # pose = self.trajectory['action']['target_cartesian_position'][i]
        # pose = self.trajectory['action']['cartesian_position'][i]

        # Link to world coordinate
        trans, mat = extract_extrinsics(pose) # [3], [3, 3]
        self.world_pos_3d = ext_to_world(trans, mat) # [4, 4]
        # END MV
        
        pose = self.trajectory['action']['cartesian_position'][i]
        trans, mat = extract_extrinsics(pose)
        rr.log('action/cartesian_position/transform', rr.Transform3D(translation=trans, mat3x3=mat))
        rr.log('action/cartesian_position/origin', rr.Points3D([trans]))

        log_cartesian_velocity('action/cartesian_velocity', self.action['cartesian_velocity'][i])

        rr.log('action/gripper_position', rr.Scalar(self.action['gripper_position'][i]))
        rr.log('action/gripper_velocity', rr.Scalar(self.action['gripper_velocity'][i]))

        for j, vel in enumerate(self.trajectory['action']['cartesian_position'][i]):
            rr.log(f'action/joint_velocity/{j}', rr.Scalar(vel))

        # === target_cartesian_position
        # pose = self.trajectory['action']['target_cartesian_position'][i]
        # trans, mat = extract_extrinsics(pose)
        # rr.log('action/target_cartesian_position/transform', rr.Transform3D(translation=trans, mat3x3=mat))
        # rr.log('action/target_cartesian_position/origin', rr.Points3D([trans]))

        # rr.log('action/target_gripper_position', rr.Scalar(self.action['target_gripper_position'][i]))
        
    def log_robot_state(self, i: int, entity_to_transform: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        
        joint_angles = self.robot_state['joint_positions'][i]
        for joint_idx, angle in enumerate(joint_angles):
            # MV
            # get 3D position for string name of robot part
            log_angle_rot(entity_to_transform, joint_idx + 1, angle)

        if i > 1:
            lines = []
            for j in range(i-1, i+1):
                # MV len(joint_angles) = 7
                joint_angles = self.robot_state['joint_positions'][j]
                joint_origins = []
                for joint_idx in range(len(joint_angles)+1):
                    transform = link_to_world_transform(entity_to_transform, joint_angles, joint_idx+1)
                    # MV
                    if joint_idx == len(joint_angles):
                        left_finger = 9
                        left_transform = link_to_world_transform(entity_to_transform, joint_angles, left_finger)
                        right_finger = 10
                        right_transform = link_to_world_transform(entity_to_transform, joint_angles, right_finger)
                        self.finger_tip = (left_transform + right_transform) / 2
                        # END MV
                    joint_org = (transform @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
                    joint_origins.append(list(joint_org))
                lines.append(joint_origins)
            
            for traj in range(len(lines[0])):
                rr.log(f"trajectory/{traj}", rr.LineStrips3D([origins[traj] for origins in lines]))
        
        rr.log('robot_state/gripper_position', rr.Scalar(self.robot_state['gripper_position'][i]))

        for j, vel in enumerate(self.robot_state['joint_velocities'][i]):
            rr.log(f"robot_state/joint_velocities/{j}", rr.Scalar(vel))

        for j, vel in enumerate(self.robot_state['joint_torques_computed'][i]):
            rr.log(f"robot_state/joint_torques_computed/{j}", rr.Scalar(vel))

        for j, vel in enumerate(self.robot_state['motor_torques_measured'][i]):
            rr.log(f"robot_state/motor_torques_measured/{j}", rr.Scalar(vel))

    def log(self, urdf_logger) -> None:
        time_stamps_nanos = self.trajectory["observation"]["timestamp"]["robot_state"][
            "robot_timestamp_nanos"
        ]
        time_stamps_seconds = self.trajectory["observation"]["timestamp"][
            "robot_state"
        ]["robot_timestamp_seconds"]
        for i in range(self.trajectory_length):
            time_stamp = time_stamps_seconds[i] * int(1e9) + time_stamps_nanos[i]
            rr.set_time_nanos("real_time", time_stamp)

            if i == 0:
                # We want to log the robot model here so that it appears in the right timeline
                urdf_logger.log()

            # MV
            self.log_robot_state(i, urdf_logger.entity_to_transform)
            self.log_action(i)
            self.log_cameras_next(i)

            if i > 600:
              break

    # MV
    def draw_image(self, path):
        plot = draw_sequence(self.imsaver.snapshots["grip"], self.points)
        io.imsave(path, plot, quality=90)

        # grip image
        Path("data/frames").mkdir(parents=True, exist_ok=True)
        io.imsave("data/frames/first_image.jpg", self.imsaver.snapshots["first"], quality=90)
        io.imsave("data/frames/grip_image.jpg", self.imsaver.snapshots["grip"], quality=90)
        io.imsave("data/frames/max_image.jpg", self.imsaver.snapshots["max"], quality=90)
        for time, image in self.imsaver.center_images:
            io.imsave(f"data/frames/center_{time:0>16}.jpg", image, quality=90)


def blueprint_raw():
    from rerun.blueprint import (
        Blueprint,
        BlueprintPanel,
        Horizontal,
        Vertical,
        SelectionPanel,
        Spatial3DView,
        TimePanel,
        TimeSeriesView,
        Tabs,
    )

    blueprint = Blueprint(
        Horizontal(
            Vertical(
                Spatial3DView(name="robot view", origin="/", contents=["/**"]),
                blueprint_row_images(
                    [
                        "cameras/ext1/left",
                        "cameras/ext1/right",
                        "cameras/ext1/depth",
                        "cameras/wrist/left",
                    ]
                ),
                blueprint_row_images(
                    [
                        "cameras/ext2/left",
                        "cameras/ext2/right",
                        "cameras/ext2/depth",
                        "cameras/wrist/right",
                    ]
                ),
                row_shares=[3, 1, 1],
            ),
            Tabs(
                Vertical(
                    *(
                        TimeSeriesView(origin=f'action/cartesian_velocity/{dim_name}') for dim_name in POS_DIM_NAMES
                    ),
                    name='cartesian_velocity',
                ),
                Vertical(
                    TimeSeriesView(origin='action/', contents=['action/gripper_position', 'action/target_gripper_position']),
                    TimeSeriesView(origin='action/gripper_velocity'),
                    name='action/gripper' 
                ),
                Vertical(
                    *(
                        TimeSeriesView(origin=f'action/joint_velocity/{i}') for i in range(6)
                    ),
                    name='action/joint_velocity'
                ),
                Vertical(
                    *(
                        TimeSeriesView(origin=f'robot_state/joint_torques_computed/{i}') for i in range(7)
                    ),
                    name='joint_torques_computed',
                ),
                Vertical(
                    *(
                        TimeSeriesView(origin=f'robot_state/joint_velocities/{i}') for i in range(7)
                    ),
                    name='robot_state/joint_velocities'
                ),
                Vertical(
                    *(
                        TimeSeriesView(origin=f'robot_state/motor_torques_measured/{i}') for i in range(7)
                    ),
                    name='motor_torques_measured',
                ),
            ),
            column_shares=[3, 2],
        ),
        BlueprintPanel(expanded=False),
        SelectionPanel(expanded=False),
        TimePanel(expanded=False),
        auto_space_views=False,
    )

    # MV
    mv_blueprint = Blueprint(
        Vertical(
            blueprint_row_images(
                [
                    "cameras/ext1/left",
                ]
            ),
            row_shares=[3, 1],
        ),
        BlueprintPanel(expanded=False),
        SelectionPanel(expanded=False),
        TimePanel(expanded=False),
        auto_space_views=False,
    )
    return mv_blueprint

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

    # remove leftover files
    for file in Path("data/frames").glob("center*jpg"):
        file.unlink()

    # args.visualize: bool
    rr.init("DROID-visualized", spawn=args.visualize) # MV

    urdf_logger = URDFLogger("franka_description/panda.urdf")

    raw_scene: RawScene = RawScene(args.scene, args.visualize)
    rr.send_blueprint(blueprint_raw())
    raw_scene.log(urdf_logger)

    # MV
    plot_dir = Path(args.plot).parent
    plot_dir.mkdir(parents=True, exist_ok=True)
    raw_scene.draw_image(args.plot)
    raw_scene.draw_image("data/plot.jpg")

    logdata = {
        "visible_points": raw_scene.visible_count,
        "gripper_closed_times": raw_scene.gripper_close_count,
        "gripper_duration": raw_scene.gripper_duration,
        "episode_duration": raw_scene.trajectory_length,
        "first_touch": list(raw_scene.first_touch_2d),
    }
    with open("data/single_log.json", "w") as f:
        json.dump(logdata, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
