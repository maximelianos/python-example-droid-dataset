# compute difference between crops

import numpy as np
from skimage import io
from skimage.transform import rescale, resize

import torch
from transformers import Dinov2Backbone


def main():
    # model_ = torch.load("weights/dinov2_s_cow_front_face_fine_tuned_model.pth")
    model = Dinov2Backbone.from_pretrained("facebook/dinov2-base")

    # pixel_values = torch.randn(1, 3, 224, 224)
    #
    # outputs = model(pixel_values)
    #
    # for feature_map in outputs.feature_maps:
    #     print(feature_map.shape)

    imginfo = lambda img: print(type(img), img.dtype, img.shape, img.min(), img.max())

    first_image = io.imread("first_image.jpg")
    max_image = io.imread("max_image.jpg")

    diff = first_image - max_image
    diff = np.sum(diff ** 2, axis=2) ** 0.5 * 10
    imginfo(diff)

    io.imsave("result.jpg", diff.astype(np.uint8))

    def to_batch(image):
        downscaled = resize(image, (224, 224))
        batch = torch.from_numpy(downscaled)[None, :, :, :]  # [h, w, c] -> [1, h, w, c]
        batch = batch.permute(0, 3, 1, 2)  # [1, h, w, c] -> [1, c, h, w]
        return batch

    def crop(image):
        """
        :param image: [h, w, c]
        """
        x, y = 900, 240
        cropw = 200
        return image[y-cropw:y+cropw, x-cropw:x+cropw]

    io.imsave("result_crop_a.jpg", crop(first_image).astype(np.uint8))
    io.imsave("result_crop_b.jpg", crop(max_image).astype(np.uint8))

    batch = to_batch(crop(first_image))
    imginfo(batch)
    dino1 = model(batch).feature_maps[0].detach()

    dino2 = model(to_batch(crop(max_image))).feature_maps[0].detach()


    imginfo(dino1)
    imginfo(dino2)

    diff = dino1 - dino2
    diff = (diff[0, :, :, :] ** 2).sum(axis=0)
    diff = diff.numpy()

    def scale(diff_image: np.array):
        low, high = np.quantile(diff, 0.1), np.quantile(diff, 0.9)
        return np.clip(diff_image / high * 255, 0, 255)

    diff = scale(diff)
    io.imsave("result.jpg", resize(diff, (400, 400)).astype(np.uint8))

if __name__ == "__main__":
    main()