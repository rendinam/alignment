#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as pp
import cv2
from astropy.io import fits

imfile = sys.argv[1]

# Take Hessian threshold from command line, if present.
# Use a sensible default otherwise.
if len(sys.argv) == 3:
    thresh = int(sys.argv[2])
else:
    thresh = 400

if imfile[-5:] == '.fits':
    print('FITS file')
    # Get image data from FITS file
    fits.info(imfile)
    
    # The actual HDU index might need to be different.
    img = fits.getdata(imfile, ext=1)
else:
    print('Non-fits image file.')
    img = cv2.imread(imfile, 0)

# Remove NaNs from array
img_nonan = np.nan_to_num(img)

# Convert float array to UINT8 array.
# This loses precision, but as long as the keypoints are valid, the
# transformation based on them can be performed on the original array.
# TODO: Does the original array need to be normalized to maximize the precision
# here?
img_nonan_int = img_nonan.astype('uint8')
print('Type : {}'.format(type(img_nonan_int)))
print('Shape: {}'.format(img_nonan_int.shape))
print(img_nonan_int)

surf = cv2.xfeatures2d.SURF_create(thresh)

# Find keypoints and descriptors
kp, des = surf.detectAndCompute(img_nonan_int, None)
print('Keypoints detected: {}'.format(len(kp)))

img2 = cv2.drawKeypoints(img_nonan_int, kp, None, (255,0,0), 4)

pp.imshow(img2, cmap='Greens')
pp.show()
