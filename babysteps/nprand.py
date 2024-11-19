import numpy as np
np.random.seed(0)

print("uniform distribution")
print(np.random.uniform(size=(5)))

# draw embedding vectors with replacement
embeddings = np.array([[10, 1], [2, 5]])
ind = np.random.randint(0, len(embeddings), size=5)
print("index", ind)
print("samples")
print(embeddings[ind, :])

# split into train and validation
ind = np.random.permutation(10)
print("train:", ind[0:7])
print("validation:", ind[7:10])