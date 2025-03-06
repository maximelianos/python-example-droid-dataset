import json
from pathlib import Path

photo_dir = Path("data/trajectory/dataset")

annotations = [
    "put the green cube onto the yellow cube",
    "put the yellow cube onto the blue cube",
    "take the blue cube and put it on the yellow cube",
    "put the green cube onto the yellow cube",
    "put the green cube onto the yellow cube",
    "put the yellow banana onto the cup",
    "put the cup onto the yellow banana",
]

i = 0
for path in sorted(photo_dir.glob("*jpg")):
    info = {
        "action_text": annotations[i],
        "camera_intrinsic": [[[524.2742919921875, 0.0, 639.7766723632812], [0.0, 524.2742919921875, 370.2777099609375], [0.0, 0.0, 1.0]]],
        "camera_extrinsic": [[[-0.756711653417621, -0.6536150850791974, 0.013220973446916529, 0.563833255544263], [-0.27160423137160716, 0.29592113880925086, -0.9157848115724976, 0.14376940963332063], [0.5946584020116162, -0.6965759112708032, -0.40145159702539657, 0.4036810414170865]]],
        "obj_start_pose": [[0.45362711345467494, 0.25685453321107765, -0.007110005493356275, 0.04105788848075122, 0.9923989787858659, 0.10455640421285424, 0.05026405312299101]],
        "obj_end_pose": [[0.4962653079377724, 0.0011648163444125878, 0.07070264414012217, 0.12667444027092917, 0.9262209299758405, -0.353080233918236, 0.037452950123851685]],
        "tcp_start_pose": [[0.3666710457032203, 0.07985811068574615, 0.4060562979142576, -0.0695148033990866, 0.9824752780862268, -0.13179048859946485, -0.11198788850805164]],
        "grasp_pose": [[0.45362711345467494, 0.25685453321107765, -0.007110005493356275, 0.04105788848075122, 0.9923989787858659, 0.10455640421285424, 0.05026405312299101]],
        "robot_pose": [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
        "image": path.name,
    }
    _path = Path("data/trajectory/_annotations.all.jsonl")
    with open(_path, "a") as f:
        line = json.dumps(info, ensure_ascii=False)
        print(line, file=f)

    i += 1
