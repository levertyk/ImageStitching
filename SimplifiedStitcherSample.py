import numpy as np
import cv2 as cv

# Hardcoded image paths
image1_path = "SampleSets/parrington/prtn12.jpg"
image2_path = "SampleSets/parrington/prtn13.jpg"
output_path = "result2.jpg"

# Read input images
image1 = cv.imread(image1_path)
image2 = cv.imread(image2_path)

# Create Stitcher object with default mode (PANORAMA)
stitcher = cv.createStitcher() 

# Stitch the images
status, pano = stitcher.stitch((image1, image2))

if status == cv.Stitcher_OK:
    # Write the stitched image to output path
    cv.imwrite(output_path, pano)
    print("Stitching completed successfully. %s saved!" % output_path)
else:
    print("Can't stitch images, error code = %d" % status)

cv.destroyAllWindows()