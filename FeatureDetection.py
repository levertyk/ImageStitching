import cv2
import numpy as np

# Read the input images
image1 = cv2.imread('SampleSets/parrington/prtn12.jpg')
image2 = cv2.imread('SampleSets/parrington/prtn13.jpg')

# Create ORB object
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors for both images
keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

# Match keypoints using FLANN matcher
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                   table_number=6,  # 6
                   key_size=12,  # 12
                   multi_probe_level=1)  # 1
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Filter matches using Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Extract corresponding keypoints
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

# Estimate the transformation matrix using RANSAC
M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

# Warp image2 to align with image1
aligned_image2 = cv2.warpPerspective(image2, M, (image1.shape[1], image1.shape[0]))

# Blend the overlapping region
overlap_width = int(image1.shape[1] / 2)
blended_image = cv2.addWeighted(
    src1=image1[:, :overlap_width],
    alpha=0.5,
    src2=aligned_image2[:, :overlap_width],
    beta=0.5,
    gamma=0
)

# Combine the images
stitched_image = cv2.hconcat([blended_image, aligned_image2[:, overlap_width:]])

# Display the stitched image
cv2.imshow('Stitched Image', stitched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()