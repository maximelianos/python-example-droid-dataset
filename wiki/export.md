# Plan export train data for Max

* raw.py - for each frame return TCP XYZ + rotation in a matrix 4x4
  * `log_cameras_next() - return_dict["finger_transform"]`
* my_loader.py.save_info() - save images; calibration and poses in json
  * first frame, grip frame, first depth. `depth_mm = depth_meters * 1000.0`
  * camera extrinsic (3, 4) and intrinsic (3, 3)
  * tcp pose (tx ty tz qw qx qy qz)
* test_real.ipynb - visualize coordinate frame

# cVLA requirements

```
accelerate==1.3.0
transformers==4.48.0
```

# Mount LMB

```
sshfs -p 2122 velikanm@lmblogin.informatik.uni-freiburg.de:/misc /data
ln -s $HOME/octagon/python-example-droid-dataset/data data
```

Datasets on LMB

```
My datasets on LMB:
/data/lmbraid19/argusm/CLUSTER/octagon/python-example-droid-dataset/data/droid_raw/
Max datasets:
/mnt/lmb/lmbraid19/argusm/datasets
```

# How to run GT markup

Visualize one trajectory
```
$ python -m droidloader.raw --visualize --sid 0
Date format: YYYY-MM-DD-HHh-MMm-SSs
Example: 2023-07-07-15h-03m-33s
```

Export data
```
$ export PYTHONPATH=$PYTHONPATH:/home/argusm/lang/RAFT/core
$ python -m droidloader.my_loader        # calls process_manuals()
Open test_real.ipynb
Log "cartesian_position" in raw.py
```

# Export variants

**Variant AUTOLab.** 5077 episodes

**Variant Autolab2.** groupfilter. Reject r"(rope|cable|towel|cloth|rubber band)". Accept “autolab” in uuid.

3000 episodes, 1 of 5 = 600 episodes

**Variant 1of5.** Reject len > 200; r"(rope|cable|towel|cloth|rubber band)".

34 000 matching episodes = 5 TB. download every 5th = 6200 episodes = 1 TB (1 episode 200 MB). GT ok for 1293

Export: `clevr-real-1of5u-initial`

**Variant 5of5** Reject len > 200; r"(rope|cable|towel|cloth|rubber band)"; r"(door|spoon|kettle|curtain|hang|pillow|fold|push|press|tissue|scoop|cook|stir|switch)"

30 000 matching episodes. download 10 000 first ones


Where is yellow block variant? 250 unchecked, 160 checked

Export: `clevr-real-block-v3`


**Variant block-2000.** REJECT len > 200; regex3 = r"(close|drawer|charger|adapter)"; regex4 = r"(rope|cable|towel|cloth|rubber band)"; regex5 = r"(door|spoon|kettle|curtain|hang|pillow|fold|push|press|tissue|scoop|cook|stir|switch)"; ACCEPT "block".

Matched 1800
