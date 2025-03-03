# Export train data for Max

* raw.py - for each frame return TCP XYZ + rotation in a matrix 4x4
  * `log_cameras_next() - return_dict["finger_transform"]`
* my_loader.py.save_info() - save images; calibration and poses in json
  * first frame, grip frame, first depth. `depth_mm = depth_meters * 1000.0`
  * camera extrinsic (3, 4) and intrinsic (3, 3)
  * tcp pose (tx ty tz qw qx qy qz)
* test_real.ipynb - visualize coordinate frame


