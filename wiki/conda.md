# Conda environment

## Basic conda commands

```
$ conda env list
$ conda env remove --name torch
$ pip show gsutil
$ pip install gsutil==
```

Add bash alias to `.bashrc`

```
alias c='conda activate pfp_env'
alias j='conda activate pfp_env && cd $HOME/octagon && jupyter notebook'
```

## [Eugenio](http://pointflowmatch.cs.uni-freiburg.de/) environment

Add env variables to `.bashrc`

```
export COPPELIASIM_ROOT=${HOME}/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

Install dependencies

```
conda create --name pfp_env python=3.10
conda activate pfp_env
bash bash/install_deps.sh
bash bash/install_rlbench.sh

# Get diffusion_policy from my branch
cd ..
git clone git@github.com:chisarie/diffusion_policy.git && cd diffusion_policy && git checkout develop/eugenio 
pip install -e ../diffusion_policy

# 3dp install
cd ..
git clone git@github.com:YanjieZe/3D-Diffusion-Policy.git && cd 3D-Diffusion-Policy
cd 3D-Diffusion-Policy && pip install -e . && cd ..

# If locally (doesnt work on Ubuntu18):
pip install rerun-sdk==0.15.1
pip install gymnasium
```

## DITTO environment

```
$ pip install torch torchvision transformers  # already present in Egenio
$ pip install -r requirements.txt
$ conda install -c conda-forge libstdcxx-ng
$ pip install -e .

# install FlowControl (see DITTO instructions)
# install RAFT

$ pip install ruamel.yaml
$ pip install git+https://github.com/facebookresearch/segment-anything.git
```

## [Rerun](https://github.com/rerun-io/python-example-droid-dataset/tree/master) environment

```
$ pip install --force-reinstall charset-normalizer==3.1.0
$ conda install scikit-image=0.19 pandas matplotlib jupyter notebook
$ pip install gsutil
$ pip install -r requirements.txt
$ pip uninstall opencv-python
$ pip uninstall pyqt5
$ conda install fastai::opencv-python-headless

$ git clone https://github.com/maximelianos/python-example-droid-dataset.git
$ git checkout dev

Type annotation support:
Python 3.8 - some support
Python 3.9 - more
Python 3.10 - even more
```

Download episode text descriptions

```
$ mkdir ../droid_raw
$ gsutil -m rsync -r -x "(.*npy)|(.*mp4)|(.*svo)|(.*h5)|(failure)|(timestamp)" gs://gresearch/robotics/droid_raw/1.0.1/  droid_raw
$ python scripts/process01.py --data ../droid_raw/
```

Filter description by regex and download videos
```
$ python scripts/my_download_raw.py --debug
```

## ZED

1. [DROID page](https://droid-dataset.github.io/droid/software-setup/host-installation.html)
2. [Download ZED](https://www.stereolabs.com/docs/installation/linux) for Ubuntu 22

```
-- skip_cuda
download all AI models? no
```

Result:
```
import pyzed.sl as sl
ImportError: version `GLIBCXX_3.4.30' not found (required by /usr/local/zed/lib/libsl_zed.so

hint: GLIBCXX 3.4.30 not found in conda environment
$ strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
$ ln <target> <linkname>; -s symbolic; -f force
$ ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ${CONDA_PREFIX}/lib/libstdc++.so.6
```
