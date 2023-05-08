import cv2
import numpy as np
import PanoramaStitch as beet

in_img = []

NUM_IMG = 14

for i in range(1, NUM_IMG + 1):
    if i < 10:
        img_scalar = cv2.imread("SampleSets\parrington\prtn0" + repr(i) + ".jpg")
    else:
        img_scalar = cv2.imread("SampleSets\parrington\prtn" + repr(i) + ".jpg")
    scale = 1 #ideally, this would be NUM_IMG, but I can't get it to work on images that small.
    in_img.append(img_scalar[::scale,::scale])
    

panoramic = in_img[0]

for i in range(1, NUM_IMG):
    temp = beet.stitch(panoramic, in_img[i])
    panoramic = temp

cv2.imsave("Panorama.png", panoramic)

