# References:
#     https://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html
#     https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html

import cv2
import numpy as np

#trim: reduce the black frame around an image
#input: image
#output: image without and black around its borders
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

def stitch(img_L, img_R):
    img1 = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)  # Convert the left image to grayscale

    img2 = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)  # Convert the right image to grayscale

    sift = cv2.ORB_create(nfeatures=2000)  # Create an instance of the SIFT feature detector
    # Find the key points and descriptors in the left and right images using SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    cv2.imshow('original_image_left_keypoints', cv2.drawKeypoints(img_L, kp1, None))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    match = cv2.BFMatcher()  # Create an instance of the Brute-Force Matcher
    matches = match.knnMatch(des1, des2, k=2)  # Perform k-nearest-neighbors matching of descriptors

    good = []  # Initialize an empty list to store "good" matches
    for m, n in matches:
        if m.distance < 0.6 * n.distance:  # Apply a distance ratio test to filter out ambiguous matches
            good.append(m)  # Add "good" matches to the list

    draw_params = dict(matchColor=(0, 255, 0),  # Set parameters for drawing matches
                    singlePointColor=None,
                    flags=2)

    img3 = cv2.drawMatches(img_L, kp1, img_R, kp2, good, None, **draw_params)  # Draw the "good" matches
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
        # Display the overlapping region of the two images
    else:
        print("Not enough matches are found - %d/%d" %
            (len(good), MIN_MATCH_COUNT))

    # Funky stuff going on right here, no bueno
    # print(M)  # Print the homography matrix M

    # Warp the perspective of the second image (img_L) using the homography matrix M
    # and create a new image with a size equal to the sum of widths of img_R and img_L
    # and the height of img_R
    # TODO: this STILL might not work as intended, but it is hard to tell.
    # The canvas for the end image is dst. The warped intersection is stored in warped_inter
    dst = cv2.warpPerspective(img_L, M, (img_R.shape[1] + img_R.shape[1], img_R.shape[0]))
    warped_inter = trim(dst).copy()

    # Copy the pixels of the original image (img_R) onto the corresponding region of the warped image (dst)
    # This effectively stitches the two images together
    # TODO: Fix the slight line at the right edge of the stitch zone
    dst[0:img_L.shape[0], 0:img_L.shape[1]] = img_L
    gap_width = img_L.shape[1] - warped_inter.shape[1]
    dst[0:img_L.shape[0],  gap_width:(gap_width + img_R.shape[1])] = img_R
    dst[0:img_L.shape[0], gap_width:img_L.shape[1]] = warped_inter

    # Display the final stitched image (dst) with the original image and the warped image blended together
    cv2.imshow("original_image_stitched.jpg", dst)

    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()  # Close all display windows
    return dst
