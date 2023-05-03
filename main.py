import cv2
import numpy as np
import PanoramaStitch as beet

in_img = []

NUM_IMG = 14

for i in range(1, NUM_IMG + 1):
    #if i < 10:
    in_img.append(cv2.imread("SampleSets\library\\" + repr(i) + ".jpg"))
    #else:
    #    in_img.append(cv2.imread("SampleSets\grail\grail" + repr(i) + ".jpg"))

panoramic = beet.stitch(in_img[0], in_img[1])

for i in range(2, NUM_IMG):
    temp = beet.stitch(panoramic, in_img[i])
    panoramic.shape = temp.shape
    panoramic = temp

cv2.imsave("Panorama.png", panoramic)

