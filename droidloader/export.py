import sys
import rerun as rr
from .my_loader import DroidLoader

if len(sys.argv) < 2:
    print("Usage: python -m droidloader.export SCENE")


scene = sys.argv[1]
rr.init("DROID-visualized", spawn=False) # MV
loader = DroidLoader(scene)
loader.read_trajectory()
# loader.track()
# loader.track_3d() # [4] = XYZ+color
# loader.read_trajectory() # raw.py will cut pcd now
# loader.save_pcd()
#

loader.save_info()
