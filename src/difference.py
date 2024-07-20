from pathlib import Path
import json

import numpy as np
from skimage import io
from skimage.transform import rescale, resize
from scipy.ndimage import gaussian_filter
from scipy import ndimage

import torch
# from transformers import Dinov2Backbone
from torchvision.models import resnet18

imginfo = lambda img: print(type(img), img.dtype, img.shape, img.min(), img.max())

class Difference:
    def __init__(self):
        self.code_dir = Path(__file__).parent.parent

        backbone = resnet18(pretrained=True)

        children = list(backbone.children())
        self.newmodel = torch.nn.Sequential(*children[0:6])

    def process(self):
        code_dir = self.code_dir

        # === load first touch coors
        with open(code_dir / "data/single_log.json") as f:
            episode_log = json.load(f)
        first_x, first_y, first_z = map(int, episode_log["first_touch"])
        cropw = 200

        # === first image VS max distant image during grip
        data_path = code_dir / "data/frames"
        file_list = sorted(data_path.glob("center*jpg"))
        first_image = io.imread(file_list[0])
        mid_image = io.imread(file_list[-1])
        max_image = io.imread(data_path / "max_image.jpg")

        h, w = first_image.shape[:2]
        if not (0 <= first_y - cropw and first_y + cropw < h and
            0 <= first_x - cropw and first_x + cropw < w):
            # bad episode
            print("bad episode")
            io.imsave(data_path / "result_overlay.jpg", first_image)
            return

        def to_batch(image):
            downscaled = resize(image, (224, 224))
            batch = torch.from_numpy(downscaled)[None, :, :, :].to(torch.float32) / 255  # [h, w, c] -> [1, h, w, c]
            batch = batch.permute(0, 3, 1, 2)  # [1, h, w, c] -> [1, c, h, w]
            return batch

        def cut_crop(image):
            """
            :param image: [h, w, c]
            """
            return image[first_y-cropw:first_y+cropw, first_x-cropw:first_x+cropw]

        save_images = {
            "crop_first": cut_crop(first_image),
            "crop_max": cut_crop(max_image)
        }
        for name, image in save_images.items():
            io.imsave(data_path / (name + ".jpg"), image.astype(np.uint8))

        # === Run resnet on crops
        dino1 = self.newmodel(to_batch(cut_crop(first_image))).detach() # [1, c, h, w] range [0, 1] torch.float32
        dino2 = self.newmodel(to_batch(cut_crop(max_image))).detach()

        def scale(diff_image: np.ndarray):
            low, high = np.quantile(diff_image, 0.1), np.quantile(diff_image, 0.99)
            return np.clip(diff_image / high * 255, 0, 255)

        diff = dino1 - dino2
        diff = (diff[0, :, :, :] ** 2).sum(axis=0).numpy() # [h, w]

        dino_mid = self.newmodel(to_batch(cut_crop(mid_image))).detach()
        diff_mid = ((dino_mid - dino2)[0, :, :, :] ** 2).sum(axis=0).numpy()
        diff_mid = gaussian_filter(diff_mid, sigma=1) * 2
        diff = diff - diff_mid

        diff = resize(diff, (cropw*2, cropw*2))

        # io.imsave(data_path / "result_diff.jpg", scale(diff).astype(np.uint8))

        def overlay_crop(canvas: np.ndarray, diff: np.ndarray):
            diff = scale(resize(diff, (cropw*2, cropw*2))).astype(np.uint8)
            diff = np.tile(diff[:, :, None], (1, 1, 3))  # 1 channel to 3 channels
            diff[:, :, 1] = 0
            diff[:, :, 2] = 0
            
            c = np.copy(canvas)
            canvas_crop = c[first_y-cropw:first_y+cropw, first_x-cropw:first_x+cropw]
            c[first_y-cropw:first_y+cropw, first_x-cropw:first_x+cropw] = canvas_crop * 0.4 + diff * 0.6 # use broadcasting
            return c

        io.imsave(data_path / "result_overlay.jpg", overlay_crop(first_image, diff))
