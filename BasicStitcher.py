import cv2
import numpy as np

# Read the input images
image1 = cv2.imread('SampleSets/Xue-Mountain-Enterance/DSC_0171.jpg')
image2 = cv2.imread('SampleSets/Xue-Mountain-Enterance/DSC_0172.jpg')

# Determine the overlapping region (assuming half of the image width as overlap)
overlap_width = int(image1.shape[1] / 2)

# Extract the corresponding points for the transformation
pts1 = np.float32([[0, 0], [overlap_width, 0], [0, image1.shape[0]]])
pts2 = np.float32([[image2.shape[1] - overlap_width, 0], [image2.shape[1],
                  0], [image2.shape[1] - overlap_width, image2.shape[0]]])

# Estimate the transformation (assuming affine transformation)
transformation_matrix = cv2.getAffineTransform(pts1, pts2)

# Apply the estimated transformation to image2
aligned_image2 = cv2.warpAffine(
    src=image2,
    M=transformation_matrix,
    dsize=(image1.shape[1], image1.shape[0])
)

# Blend the overlapping region
blended_image = cv2.addWeighted(
    src1=image1[:, :overlap_width],
    alpha=0.5,
    src2=aligned_image2[:, :overlap_width],
    beta=0.5,
    gamma=0
)

# Combine the images
stitched_image = cv2.hconcat(
    [blended_image, aligned_image2[:, overlap_width:]])

# Original Images
cv2.imshow('Original', np.concatenate([image1, image2], axis=1))

# Display the stitched image
cv2.imshow('Stitched Image', stitched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
