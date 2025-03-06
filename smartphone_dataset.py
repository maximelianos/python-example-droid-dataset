import json
from pathlib import Path

photo_dir = Path("data/photos")

for path in sorted(photo_dir.glob("*jpg")):
    info = {
        "action_text": "put the green cube onto the red cube",
        "camera_extrinsic": [
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        ],
        "camera_intrinsic": [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ],
        "obj_start_pose": [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
        "obj_end_pose": [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
        "robot_pose": [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
        "tcp_start_pose": [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
        "grasp_pose": [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
        "image": path.name,
    }
    _path = Path("data/trajectory/_annotations.all.jsonl")
    with open(_path, "a") as f:
        line = json.dumps(info, ensure_ascii=False)
        print(line, file=f)
