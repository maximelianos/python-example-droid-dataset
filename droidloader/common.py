#!/usr/bin/env python3

import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation
import h5py

CAMERA_NAMES = ["ext1", "ext2", "wrist"]

# Maps indices in X_position and X_velocity to their meaning.
POS_DIM_NAMES = ["x", "y", "z", "rot_x", "rot_y", "rot_z"]


def link_to_world_transform(
    entity_to_transform: dict[str, tuple[np.ndarray, np.ndarray]],
    joint_angles: list[float],
    link: int,
) -> np.ndarray:

    """
    :return: [4, 4] transform matrix
    """

    # MV
    # import json
    # print(json.dumps(list(entity_to_transform.keys()), indent=4))
    # input("continue")

    # len(joint_angles) == 7
    # link = 1, ..., 8

    # print("len(joint_angles)", len(joint_angles))
    # print("link", link)

    return_left_finger_tip: bool = False
    return_right_finger_tip: bool = False
    if link == 9:
        return_left_finger_tip = True
        link = 8
    elif link == 10:
        return_right_finger_tip = True
        link = 8
    # END MV

    tot_transform = np.eye(4)
    for i in range(1, link+1):
        entity_path = path_to_link(i)
        # print("i", i)
        # print("entity_path", entity_path)

        start_translation, start_rotation_mat = entity_to_transform[entity_path]

        if i-1 >= len(joint_angles):
            angle_rad = 0
        else:
            angle_rad = joint_angles[i-1]
        vec = np.array(np.array([0, 0, 1]) * angle_rad)
        
        rot = Rotation.from_rotvec(vec).as_matrix()
        rotation_mat = start_rotation_mat @ rot
        
        transform = np.eye(4)
        transform[:3,:3] = rotation_mat
        transform[:3,3] = start_translation
        tot_transform = tot_transform @ transform

    # MV
    if return_left_finger_tip:
        # entity_path = "panda_link0/panda_link1/panda_link2/panda_link3/panda_link4/panda_link5/panda_link6/panda_link7/panda_link8/robotiq_85_base_link/left_inner_knuckle/left_inner_finger"
        # i = 1, ..., link
        # link = 1, ..., 8

        appendix = "robotiq_85_base_link/left_inner_knuckle/left_inner_finger".split("/")
        link_entity_path = path_to_link(8)
        for i in range(len(appendix)):
            entity_path = link_entity_path + "/" + "/".join(appendix[:i+1])
            #print(entity_path)

            start_translation, start_rotation_mat = entity_to_transform[entity_path]

            angle_rad = 0
            vec = np.array(np.array([0, 0, 1]) * angle_rad)
            
            rot = Rotation.from_rotvec(vec).as_matrix()
            rotation_mat = start_rotation_mat @ rot
            
            transform = np.eye(4)
            transform[:3,:3] = rotation_mat
            transform[:3,3] = start_translation
            tot_transform = tot_transform @ transform
    #input("continue")

    if return_right_finger_tip:
        appendix = "robotiq_85_base_link/right_inner_knuckle/right_inner_finger".split("/")
        link_entity_path = path_to_link(8)
        for i in range(len(appendix)):
            entity_path = link_entity_path + "/" + "/".join(appendix[:i+1])
            #print(entity_path)

            start_translation, start_rotation_mat = entity_to_transform[entity_path]

            angle_rad = 0
            vec = np.array(np.array([0, 0, 1]) * angle_rad)
            
            rot = Rotation.from_rotvec(vec).as_matrix()
            rotation_mat = start_rotation_mat @ rot
            
            transform = np.eye(4)
            transform[:3,:3] = rotation_mat
            transform[:3,3] = start_translation
            tot_transform = tot_transform @ transform

    return tot_transform

def log_cartesian_velocity(root: str, cartesian_velocity: np.ndarray):
    if not root.endswith("/"):
        root += "/"

    for vel, name in zip(cartesian_velocity, POS_DIM_NAMES):
        rr.log(root + name, rr.Scalar(vel))


def blueprint_row_images(origins):
    from rerun.blueprint import Horizontal, Spatial2DView

    return Horizontal(
        *(
            Spatial2DView(
                name=org,
                origin=org,
            )
            for org in origins
        ),
    )


def extract_extrinsics(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Takes a vector with dimension 6 and extracts the translation vector and the rotation matrix"""
    translation = pose[:3]
    rotation = Rotation.from_euler("xyz", np.array(pose[3:])).as_matrix()
    return (translation, rotation)


def path_to_link(link: int) -> str:
    return "/".join(f"panda_link{i}" for i in range(link + 1))


def log_angle_rot(
    entity_to_transform: dict[str, tuple[np.ndarray, np.ndarray]],
    link: int,
    angle_rad: float,
) -> None:
    """Logs an angle for the franka panda robot"""
    entity_path = path_to_link(link)

    start_translation, start_rotation_mat = entity_to_transform[entity_path]

    # All angles describe rotations around the transformed z-axis.
    vec = np.array(np.array([0, 0, 1]) * angle_rad)

    rot = Rotation.from_rotvec(vec).as_matrix()
    rotation_mat = start_rotation_mat @ rot

    # left_inner_finger: str
    # right_inner_finger: str
    # for name in entity_to_transform:
    #     if "left_inner_finger" in name:
    #         left_inner_finger = name
    #     if "right_inner_finger" in name:
    #         right_inner_finger = name

    rr.log(
        entity_path, rr.Transform3D(translation=start_translation, mat3x3=rotation_mat)
    )


def h5_tree(val, pre=""):
    # Taken from https://stackoverflow.com/questions/61133916/is-there-in-python-a-single-function-that-shows-the-full-structure-of-a-hdf5-fi
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            # the last item
            if type(val) == h5py._hl.group.Group:
                print(pre + "└── " + key)
                h5_tree(val, pre + "    ")
            else:
                print(pre + "└── " + key + " (%d)" % len(val))
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + "├── " + key)
                h5_tree(val, pre + "│   ")
            else:
                print(pre + "├── " + key + " (%d)" % len(val))
