import numpy as np

a = np.array([[10, 1], [2, 5]])
print("stack")
assert np.stack([a[0], a[1]]).shape == (2, 2)

print("pad")
pad_width = ((0, 2), (0, 0))
assert np.pad(a, pad_width, mode="edge").shape == (4, 2)
