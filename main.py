import cv2
import numpy as np
import PanoramaStitch as beet

left_img = cv2.imread("left.jpg")
right_img = cv2.imread("right.jpg")

panoramic = beet.stitch(left_img, right_img)
