import numpy as np
import matplotlib.pyplot as plt

a=np.load("img00000000.npy")
fig = plt.figure
plt.imshow(a, cmap='gray')
plt.show()

print(a.shape)
