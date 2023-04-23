import cv2
import numpy as np

img_ = cv2.imread('left.jpg')  # Read the left image
img_ = cv2.resize(img_, (0, 0), fx=1, fy=1)  # Resize the left image (no scaling)
img1 = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)  # Convert the left image to grayscale

img = cv2.imread('right.jpg')  # Read the right image
img = cv2.resize(img, (0, 0), fx=1, fy=1)  # Resize the right image (no scaling)
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the right image to grayscale

sift = cv2.SIFT_create()  # Create an instance of the SIFT feature detector
# Find the key points and descriptors in the left and right images using SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

cv2.imshow('original_image_left_keypoints', cv2.drawKeypoints(img_, kp1, None))
cv2.waitKey(0)
cv2.destroyAllWindows()

match = cv2.BFMatcher()  # Create an instance of the Brute-Force Matcher
matches = match.knnMatch(des1, des2, k=2)  # Perform k-nearest-neighbors matching of descriptors

good = []  # Initialize an empty list to store "good" matches
for m, n in matches:
    if m.distance < 0.03 * n.distance:  # Apply a distance ratio test to filter out ambiguous matches
        good.append(m)  # Add "good" matches to the list

draw_params = dict(matchColor=(0, 255, 0),  # Set parameters for drawing matches
                   singlePointColor=None,
                   flags=2)

img3 = cv2.drawMatches(img_, kp1, img, kp2, good, None, **draw_params)  # Draw the "good" matches
cv2.imshow("original_image_drawMatches.jpg", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()


MIN_MATCH_COUNT = 10  # Minimum number of matches required for successful stitching

if len(good) > MIN_MATCH_COUNT:
    # Extract source and destination points from good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Find homography matrix using RANSAC algorithm
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h, w = img1.shape  # Get height and width of the first image
    # Define four corner points of the first image
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]
                     ).reshape(-1, 1, 2)
    # Apply perspective transformation to the corner points using the homography matrix
    dst = cv2.perspectiveTransform(pts, M)
    # Draw a polygon around the transformed corner points on the second image
    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    # Display the stitched image with the overlapping region highlighted
    cv2.imshow("original_image_overlapping.jpg", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Not enough matches are found - %d/%d" %
          (len(good), MIN_MATCH_COUNT))

# Funky stuff going on right here, no bueno
print(M)  # Print the homography matrix M

# Warp the perspective of the second image (img_) using the homography matrix M
# and create a new image with a size equal to the sum of widths of img and img_
# and the height of img
dst = cv2.warpPerspective(img_, M, (img.shape[1] + img_.shape[1], img.shape[0]))

# Display the warped image (dst) without blending with the original image (img)
cv2.imshow("og_dst", dst)

# Copy the pixels of the original image (img) onto the corresponding region of the warped image (dst)
# This effectively stitches the two images together
dst[0:img.shape[0], 0:img.shape[1]] = img

# Display the final stitched image (dst) with the original image and the warped image blended together
cv2.imshow("original_image_stitched.jpg", dst)

cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()  # Close all display windows



def trim(frame):
    # crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    # crop top
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    # crop top
    if not np.sum(frame[:, 0]):
        return trim(frame[:, 1:])
    # crop top
    if not np.sum(frame[:, -1]):
        return trim(frame[:, :-2])
    return frame


cv2.imshow("original_image_stitched_crop.jpg", trim(dst))
#cv2.imsave("original_image_stitched_crop.jpg", trim(dst))
cv2.waitKey(0)
cv2.destroyAllWindows()
