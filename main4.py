import cv2
import numpy as np
import PanoramaStitch as beet

in_img = []

NUM_IMG = 6

for i in range(1, NUM_IMG + 1):
    img_scalar = cv2.imread("SampleSets\Xue-Mountain-Enterance\DSC_017" + repr(i) + ".jpg")
    
    scale = 1 #ideally, this would be NUM_IMG, but I can't get it to work on images that small.
    in_img.append(img_scalar[::scale,::scale])
    
panoramic = in_img[0]

for i in range(1, NUM_IMG):
    temp = beet.stitch(panoramic, in_img[i])
    panoramic = temp

cv2.imsave("Panorama.png", panoramic)

