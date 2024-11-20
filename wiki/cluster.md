# Cluster training


Home: `/misc/lmbraid19/argusm/CLUSTER/octagon`

Data for Cluster

```
data / 
  existing_episodes.json
  data/droid_raw/1.0.1/annotations.json
  manual_episodes.json <--- important
  trajectory / 2023-03-02-15h-14m-31s_traj3d.npy <--- important
```

Login by sssh

```
ssh velikanm@lmblogin.informatik.uni-freiburg.de -p 2122
ssh lmbtorque
cd octagon/PointFlowMatch
qsub -q student scripts/train.sh
python scripts/train.py log_wandb=False dataloader.num_workers=0 task_name=unplug_charger +experiment=pointflowmatch_so3
```

Submit command: `qsub -l nodes=1:ppn=4:gpus=1,mem=8000,walltime=20:00:00 -q student SCRIPTNAME`

## Preparation

1. `pip install --force-reinstall "huggingface_hub<0.26"`
2. wandb login
