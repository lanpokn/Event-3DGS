import numpy as np
import cv2
# Specify the path to the OpenCV library
opencv_path = "C:/Users//Desktop/code/opencv/build/x64/vc15/bin"
import os
# Add the library path

import sgrb2lum

# Load an SRGB image
srgb = cv2.imread("image.png", cv2.IMREAD_COLOR)
srgb = srgb.astype(np.float32)

# Convert SRGB to luminance
luminance = sgrb2lum.SGRB2Luminance(srgb)

print(luminance)