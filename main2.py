import cv2
import numpy as np
import PanoramaStitch as beet

# Load images
img1 = cv2.imread('left.jpg') 
img2 = cv2.imread('right.jpg')

scale=1
panoramic = beet.stitch(img1[::scale,::scale], img2[::scale,::scale])

cv2.imshow("Panorama.png", panoramic)


