# Implementation and testing

## Run

Compute trajectory DITTO + rerun
```
export PYTHONPATH=$PYTHONPATH:/home/argusm/lang/RAFT/core
python -m droidloader.my_loader --sid 0
python -m droidloader.raw --visualize --sid 0
```

Eugenio eval
```
python scripts/evaluate.py log_wandb=False env_runner.env_config.vis=True policy.ckpt_name=1717446544-didactic-woodpecker
```

Eugenio training (to run with DROID data edit `pfp/data/dataset_pcd.py`)
```
python scripts/train.py log_wandb=False dataloader.num_workers=0 task_name=unplug_charger +experiment=pointflowmatch_so3
```




## Dataloader implementation

`my_loader.py`
* Select episodes manually by deleting pictures, save to `data/manual_episodes.json` in format `2023-10-27-19h-48m-17s`
* Cache object box and segmentation. Key: episode date
* Cache computed trajectory

### Run

```
$ python -m droidloader.my_loader --scene data/droid_raw/1.0.1/success/2023-10-27/Fri_Oct_27_19:48:17_2023
```

## Episode selection

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

The result trajectory is `[4, 4]`: `[:3, :3]` rotation + `[:3, 3]` translation relative to camera origin.
```
```

## Rerun

Visualize

```
Episode index to episode localpath
$ python my_plot_everything.py

Visualize an episode
$ python -m droidloader.raw --visualize --scene data/droid_raw/1.0.1/success/2023-10-27/Fri_Oct_27_19:48:17_2023

Compute statistics for all episodes in process04.ipynb
```
