import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# %matplotlib inline


ar = np.arange(1,13)

print(ar.reshape(2,2,3))

print("after reshape:" + str(ar.reshape(4,3)))

print("after reshape2:" + str(ar.reshape(3,-1).T))


zeros = np.zeros((2,1))
print(zeros)


print(np.sum(ar))