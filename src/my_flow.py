# hint: torchvision optical flow
# Find the approximate TCP position.
# I take the first frame during grip and the next frame, compute optical flow,
# and take the lowest part in the moving field.
# The optical flow is too inaccurate for this task, approximation is bad.

from pathlib import Path
import numpy as np
import skimage
from skimage import io
from skimage.transform import rescale, resize
from scipy.ndimage import gaussian_filter
from scipy import ndimage

import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torchvision.transforms as T

imginfo = lambda img: print(type(img), img.dtype, img.shape, img.min(), img.max())

from torchvision.models.optical_flow import raft_large



def draw_sequence(image: np.array, points: list):
    """
    :param points: [(x, y, color)]
        color presets: 0 - begin green, end red;
        > 0 - fixed colors.
    """
    colors = {}
    colors[0] = np.array([255, 0, 0])
    colors[1] = np.array([0, 255, 0])
    colors[2] = np.array([0, 0, 255])

    canvas = np.copy(image)
    for i, (x, y, color) in enumerate(points):
        print(x, y)
        rows, cols = skimage.draw.disk((y, x), 8, shape=canvas.shape)
        k = i / len(points)

        if color == 0:
            # mix
            canvas[rows, cols] = colors[1] * (1-k) + colors[0] * k
        else:
            # fixed color
            canvas[rows, cols] = colors[color]

    return canvas


class FlowProcessor:
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent / "data"

        self.device = "cuda"
        self.model = raft_large(pretrained=True, progress=False).to(self.device).eval()
    
    def process(self):
        # === load first touch coors
        import json
        with open(self.data_dir / "single_log.json") as f:
            episode_log = json.load(f)
        first_x, first_y, first_z = map(int, episode_log["first_touch"])
        cropw = 200


        # === read numpy image
        def read_image(path):
            return io.imread(path).astype(np.float32) / 255


        # === prepare batch for RAFT
        def to_batch(image):
            batch = torch.from_numpy(image)[None, :, :, :].to(torch.float32) / 255  # [h, w, c] -> [1, h, w, c]
            batch = batch.permute(0, 3, 1, 2)  # [1, h, w, c] -> [1, c, h, w]
            
            transforms = T.Compose(
                [
                    T.ConvertImageDtype(torch.float32),
                    T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
                    #T.Resize(size=(520, 960)),
                ]
            )
            batch = transforms(batch)
            return batch


        # === compute flow with RAFT
        from torchvision.utils import flow_to_image

        def compute_flow_picture(img1_batch, img2_batch):
            with torch.no_grad():
                list_of_flows = self.model(img1_batch.to(self.device), img2_batch.to(self.device))
            print(f"length = {len(list_of_flows)} = number of iterations of the model")
            predicted_flows = list_of_flows[-1].detach()

            flow_imgs = flow_to_image(predicted_flows)
            
            # The images have been mapped into [-1, 1] but for plotting we want them in [0, 1]
            img1_batch = [(img1 + 1) / 2 for img1 in img1_batch]
            
            grid = [[img1, flow_img] for (img1, flow_img) in zip(img1_batch, flow_imgs)]
            
            return grid[0][1]


        # === save image from RAFT
        def save_torch(path, tensor):
            """
            tensor: [c, h, w] cuda float32 [0, 1]
            """
            img = (tensor.to("cpu").permute([1, 2, 0]).numpy() * 255).astype(np.uint8)
            io.imsave(path, img)


        # === run flow
        file_list = sorted(Path(self.data_dir / "frames").glob("center*jpg"))
        i1 = read_image(file_list[2]) # [h, w, 3] float32 [0, 1]
        i2 = read_image(self.data_dir / "frames/grip_image.jpg")

        # io.imsave("data/image_1.jpg", (i1 * 255).astype(np.uint8))
        # io.imsave("data/image_2.jpg", (i2 * 255).astype(np.uint8))

        img1_batch = to_batch(i2)
        img2_batch = to_batch(i1)
        # flow = compute_flow_picture(img1_batch, img2_batch)
        # save_torch("data/raft_flow.jpg", flow)

        # === segment flow

        # idea is to take the moving region closest to the center.
        # For that I first threshold the moving part,
        # then take histogram of distances from the center.
        def compute_flow(img1_batch, img2_batch):
            with torch.no_grad():
                list_of_flows = self.model(img1_batch.to(self.device), img2_batch.to(self.device))
            print(f"length = {len(list_of_flows)} = number of iterations of the model")
            predicted_flows = list_of_flows[-1].detach().cpu()

            return predicted_flows
        
        flow = compute_flow(img1_batch, img2_batch) # (u, v) - horizontal, vertical

        # magnitude
        mag = (flow[0, 0, :, :] ** 2 + flow[0, 1, :, :] ** 2) ** 0.5

        def to_01(image):
            # image: numpy float32 [a, b]
            low, high = np.quantile(image, 0.01), np.quantile(image, 0.99)
            return np.clip(image / high, 0, 1)


        mask = to_01(mag.to("cpu").numpy())
        # io.imsave(self.data_dir / "mask.jpg", (mask * 255).astype(np.uint8))

        # === visualize
        def overlay_mask(canvas: np.ndarray, mask: np.ndarray):
            # canvas: [h, w, 3] float32 [0, 1]
            # mask: [h, w] int [0, 1]
            mask = np.tile(mask[:, :, None], (1, 1, 3))  # 1 channel to 3 channels
            mask[:, :, 1] = 0
            mask[:, :, 2] = 0
            
            c = np.copy(canvas)
            c =  c * 0.4 + mask * 0.6
            return c

        overlay = overlay_mask(i2, mask)
        io.imsave(self.data_dir / "overlay.jpg", (overlay * 255).astype(np.uint8))

        # === applying quantile to the image

        _b, _c, h, w = flow.shape
        rows = torch.arange(0, h)
        cols = torch.arange(0, w)
        grid_y, grid_x = torch.meshgrid(rows, cols, indexing='ij')
        grid_y = grid_y / h
        grid_x = grid_x / w

        selection = grid_y[ (grid_x > 0.25) & (grid_x < 0.75) & (mag > 10)] # CAN BE EMPTY
        want_area = 60 * 60
        selection_area = len(selection)
        if selection_area < want_area * 2:
            # cancel process
            return
            
        q = torch.tensor([(selection_area - want_area * 0.5) / selection_area])
        y_bound = torch.quantile(selection, q)

        # y-coordinate
        y_pix = int(((y_bound + 0.05) * h).flatten())

        # x-coordinate
        selection = grid_x[(grid_x > 0.25) & (grid_x < 0.75) & (mag > 10) & (grid_y > y_bound)] # CAN BE EMPTY
        if len(selection) < want_area * 0.25:
            # cancel process
            return

        x_bound = torch.mean(selection)
        x_pix = int((x_bound * w).flatten())

        canvas = draw_sequence(overlay * 255, [(x_pix, y_pix, 0)])
        io.imsave(self.data_dir / "overlay.jpg", canvas.astype(np.uint8))