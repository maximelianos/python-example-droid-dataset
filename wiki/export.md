# Export train data for Max

Save minimum data for well calibrated episodes:
* first frame,
* camera extrinsic (3, 4) and intrinsic (3, 3),
* tcp pose at grip (tx ty tz + qw qx qy qz)

We get: 2D position of object on the first frame.

## How to implement

1. raw.py: return TCP XYZ + rotation for each frame (in fact, matrix 4x4)
  `log_cameras_next - return_dict["finger_transform"]`
2. my_loader.py: save grip TCP
3. max_loader.py: load grip TCP
4. test_real.ipynb: visualize coordinate system
5. my_episode_list.py: use an episode with good calibration instead of data/manual_episodes.json

Data format:

1. data/trajectory/date_img0.jpg
2. data/trajectory/date_intrinsic.npy
3. data/trajectory/date_tcp0.npy

