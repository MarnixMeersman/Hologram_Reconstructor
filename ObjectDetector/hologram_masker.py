"""
Created on Wed Apr  8 15:04:00 2020
Path: ObjectDetector/hologram_masker.py
Author: Marnix Meersman
Email: m.f.l.meersman@student.tudelft.nl
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border
from skimage import measure
from skimage.measure import label, regionprops
from scipy import ndimage as ndi
from scipy.ndimage import measurements, center_of_mass, binary_dilation, zoom
import plotly.graph_objects as go

# Load Data
data_path = r'/reconstructions/20231212-140735/reconstructed_holograms.npy'
img = np.load(data_path)

# Visualize Data
plt.pcolormesh(img[50], cmap='Greys_r')


# Create mask

mask = np.vectorize(clear_border, signature='(n,m)->(n,m)')(img)
plt.pcolormesh(mask[70])
plt.show()