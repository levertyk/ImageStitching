import cv2
import numpy as np

# Load images
img1 = cv2.imread('left.jpg', 0)  # Load the left image in grayscale
img2 = cv2.imread('right.jpg', 0)  # Load the right image in grayscale

# Create an instance of the ORB feature detector
orb = cv2.ORB_create()

# Find the key points and descriptors in the left and right images using ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Create an instance of the FLANN Matcher
flann = cv2.FlannBasedMatcher()
# Perform k-nearest-neighbors matching of descriptors
matches = flann.knnMatch(des1, des2, k=2)

# Apply a distance ratio test to filter out ambiguous matches
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Draw the "good" matches
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=None,
                   flags=2)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)

# Display the "good" matches
cv2.imshow("original_image_drawMatches.jpg", img3)

MIN_MATCH_COUNT = 10  # Minimum number of matches required for successful stitching

if len(good_matches) > MIN_MATCH_COUNT:
    # Extract source and destination points from "good" matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography matrix using RANSAC algorithm
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h, w = img1.shape  # Get height and width of the first image
    # Define four corner points of the first image
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

    # Warp the first image to the perspective of the second image
    warped_img1 = cv2.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))

    # Overlay the warped image onto the second image
    stitched_img = cv2.addWeighted(warped_img1, 0.5, img2, 0.5, 0)

    cv2.imshow("stitched_image.jpg", stitched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Not enough matches found. Try using better input images.")
