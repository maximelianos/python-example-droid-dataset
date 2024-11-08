import argparse
from pathlib import Path
from .my_episode_list import manual_paths
from .my_loader import EpisodeList

imginfo = lambda img: print(type(img), img.dtype, img.shape, img.min(), img.max())

def main():
   # === Test EpisodeList
    eplist = EpisodeList()
    sample = eplist[0]
    print("rgb batch", end=" ")
    imginfo(sample["images"])
    print("robot state batch", end=" ")
    imginfo(sample["robot_state"])
    print("pcd")
    imginfo(sample["pcd_xyz"])

if __name__ == "__main__":
    main()
 
