# Implementation and testing

## Run

Compute trajectory DITTO + rerun
```
export PYTHONPATH=$PYTHONPATH:/home/argusm/lang/RAFT/core
python -m droidloader.my_loader --sid 0
python -m droidloader.raw --visualize --sid 0
```

## Eugenio

Test npy-only dataloader: `python -m droidloader.train_loader`

### Configuration

```
conf/train.yaml
n_obs_steps: 2
n_pred_steps: 32
subs_factor: 1, 3
batch_size: 64, 128
```

### Plot evaluation

```
- pfp/data/dataset_pcd
  batchsize = 1
- train_loader
  is_test = True
python scripts/evaluate.py log_wandb=False env_runner.env_config.vis=False policy.ckpt_name=1731491805-acoustic-bee
python -m droidloader.draw_trajectory
```

Implementation:
```
EvalResults - global buffer...

robot_state: numpy (10)
obs: (4096, 3)
predict: numpy (51, 32, 10) - (K, pred_steps, robot_state)
```

Project 3D to image:

```
point_3d [X, Y, Z, 1] (4, 1)
pinhole = np.eye(4)[:3, :4]
intrinsic (3, 3)
point_2d = intrinsic @ pinhole @ point_3d # homogenous (3, 1)
point_2d = point_2d / point_2d[2] # [x/z, y/z, 1] = [x, y, z]
x, y = point_2d[0], point_2d[1]
```

### Plan

1. Run Eugenio evaluation at `pfp / envs / rlbench_runner.py`
2. Use `eval_logger` to save trajectory to npy
3. Plot all with `python -m droidloader.draw_trajectory`

### Eugenio training

(to run with DROID data edit `pfp/data/dataset_pcd.py`)
```
python scripts/train.py log_wandb=False dataloader.num_workers=0 task_name=unplug_charger +experiment=pointflowmatch_so3
```

* Use existing checkpoint: `policy.ckpt_name=1731428232-encouraging-basilisk `
* Continue training: `+run_name=1731491805-acoustic-bee`
* Log: `pfp/policy/so3 - logger.log_metrics loss/train/xyz`

Model arch: `pfp / policy / fm_so3_policy`

Implemented ReplayBuffer

```
pcd [n_steps, n_points, 3]
step_start:step_start+MAX_STEPS + padding by same point until MAX_STEPS
```

### Plan

1. Input: sequence of rgb [batch, n, c, h, w] + seq of points [batch, n, 2] (x, y), later - object mask, text
2. Output: 2D point in next 30 frames [batch, 30, 2] (x, y)

## Single loader

`my_loader.py`
* Select episodes manually by deleting pictures, save to `data/manual_episodes.json` in format `2023-10-27-19h-48m-17s`
* Cache object box and segmentation. Key: episode date
* Cache computed trajectory

### Run

```
$ python -m droidloader.my_loader --scene data/droid_raw/1.0.1/success/2023-10-27/Fri_Oct_27_19:48:17_2023
```

## Episode selection

```
Variant Marker
filter "marker"
- reject if
  len > 60 or
  r"(take|remove|from).*(cup|mug|pot|bowl)" or
  r"move.*(forward|backwards|left|right)"

Variant Block
filter "block"
- reject if 
  len > 60 or
  r"(close|drawer)"
```

Good examples
* ind 34098, local ind 810, AUTOLab+0d4edc83+2023-10-27-19h-48m-17s
* GuptaLab+553d1bd5+2023-05-19-10h-37m-18s | Put the orange block on top of the green block | 60 episodes
* AUTOLab+84bd5053+2023-08-17-17h-02m-12s | Put the yellow block in the cup | 100 episodes!

Bad examples
* RAIL+d027f2ae+2023-06-15-12h-24m-53s | Put the green block behind the orange one
* Unstack the four blocks on the right
* RAIL+80edfcb1+2023-06-30-15h-37m-23s | Move the yellow block to the left
* TRI+938130c4+2023-08-08-09h-52m-26s | Transfer the blocks from the box to the storage unit
* TRI+938130c4+2023-08-09-16h-51m-09s | Use the chopsticks to stir the blocks in the wooden box. 
* AUTOLab+t3d58310+2023-08-12-18h-07m-13s | Put all the building blocks on the table into the black bowl

Unsure
* RAIL+d027f2ae+2023-10-09-11h-05m-13s | Put the yellow block in the red bowl 
* AUTOLab+5d05c5aa+2023-10-14-21h-42m-30s | Put the yellow, blue, red and green lego bricks in the bowl

### Data structure

```
--- data/manual_episodes.json
date "2023-03-02-15h-14m-31s"
uuid "IRIS+ef107c48+2023-03-02-15h-14m-31s"
path IRIS/success/(date)/(time)
```



## DITTO

```
$ export PYTHONPATH=$PYTHONPATH:/home/argusm/lang/RAFT/core
```

Integrating DITTO into Rerun

1. Take notebook as base: notebooks/imitation_flow_nick.ipynb
2. Data loading: DITTO.trajectory - Trajectory - from_hands23
3. Implement my own class with methods: get_timestamps, get_rgb, get_object_mask
4. Track object with Trajectory.trajectory_2D

### Plot the tracked trajectory

1. Load episode frames. `raw.py; log_cameras_next(i)`
2. Detect and segment object. `my_loader.py; my_sam.py`
3. Track. `execute DITTO -> data/trajectory.npy`
4. Visualize. `raw.py`

Debug points
* No object detected. my_sam.py ok, my_loader.py?
* Gripper not closed. Not ok!

### Run .trajectory_3D

The result trajectory is [4, 4] - [:3, :3] rotation + [:3, 3] translation relative to camera origin.

## Rerun

Visualize

```
Episode index to episode localpath
$ python my_plot_everything.py

Visualize an episode
$ python -m droidloader.raw --visualize --scene data/droid_raw/1.0.1/success/2023-10-27/Fri_Oct_27_19:48:17_2023

Compute statistics for all episodes in process04.ipynb
```
