# Development and testing

## DITTO

```
$ export PYTHONPATH=$PYTHONPATH:/home/argusm/lang/RAFT/core
```

Integrating DITTO into Rerun

1. Take notebook as base: notebooks/imitation_flow_nick.ipynb
2. Data loading: DITTO.trajectory - Trajectory - from_hands23
3. Implement my own class with methods: get_timestamps, get_rgb, get_object_mask
4. Track object with Trajectory.trajectory_2D


## Rerun

Visualize

```
Episode index to episode localpath
$ python my_plot_everything.py

Visualize an episode
$ python -m droidloader.raw --visualize --scene data/droid_raw/1.0.1/success/2023-10-27/Fri_Oct_27_19:48:17_2023

Compute statistics for all episodes in process04.ipynb
```
