# This script shows, how to
# draw a disk with skimage.draw

import numpy as np
from skimage import data
from skimage import io

from skimage.draw import disk

# hint: skimage draw - edges and lines - shapes

# Load an example image
image = data.moon()

imginfo = lambda img: print(type(img), img.dtype, img.shape, img.min(), img.max())
imginfo(image)

rows, cols = disk((200, 200), 10, shape=image.shape)
image[rows, cols] = 255

io.imsave("example.jpg", image, quality=100)
