# This script shows, how to
# print shape, type, min and max value of numpy array

import numpy as np
from skimage import data
from skimage import io

# hint: find sample images at "skimage contrast adjustment"

# Load an example image
image = data.moon()

imginfo = lambda img: print(type(img), img.dtype, img.shape, img.min(), img.max())
imginfo(image)

image[:20, :20] = 0
io.imsave("example.jpg", image, quality=100)
