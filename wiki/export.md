# Export train data for Max

Save minimum data for well calibrated episodes:
* first frame,
* camera extrinsic (3, 4) and intrinsic (3, 3),
* tcp pose at grip (tx ty tz + qw qx qy qz)

We get: 2D position of object on the first frame.

* raw.py: return TCP XYZ + rotation for each frame (in fact, matrix 4x4)
* my_loader.py: save grip TCP
* max_loader.py: load grip TCP

