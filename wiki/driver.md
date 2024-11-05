# Install CUDA driver + toolkit

```
MAX-PC specs
VGA compatible controller: NVIDIA Corporation GP104 [GeForce GTX 1070] (rev a1)
product: Intel(R) Core(TM) i5-10400F CPU @ 2.90GHz
capacity: 4300MHz
```

### Check GPU
```
$ lspci | grep -i nvidia
$ sudo lshw -C display
$ sudo ubuntu-drivers devices
```

### Install from Web

CUDA 12.2 - CUDA Toolkit: https://developer.nvidia.com/cuda-downloads

CUDA Toolkit archive: https://developer.nvidia.com/cuda-toolkit-archive

MUST PURGE BEFORE INSTALLING!

https://www.thomas-krenn.com/de/wiki/CUDA_Installation_unter_Ubuntu
1. uninstall current driver
2. install driver with apt

```
$ sudo apt install nvidia-driver-555
current: CUDA 12.2
$ sudo apt search cuda-toolkit
$ sudo apt install cuda-toolkit-12-5
```

