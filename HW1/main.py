import cv2
import numpy as np

# Load the grayscale image
image = cv2.imread('im1.jpg', cv2.IMREAD_GRAYSCALE)

# Calculate histogram
histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

# Normalize the histogram
histogram_norm = histogram.ravel() / histogram.sum()

# Compute the cumulative distribution function (CDF)
cdf = histogram_norm.cumsum()

# Initialize variables
best_threshold = 0
best_intra_class_variance = 0

# Iterate over all possible threshold values
for t in range(256):
    # Background and foreground class probabilities
    prob_background = cdf[t]
    prob_foreground = 1 - prob_background

    # Background and foreground class means
    epsilon = 1e-8  # small epsilon value to avoid division by zero
    mean_background = (histogram_norm[:t+1] * np.arange(t+1)).sum() / max(prob_background, epsilon)
    mean_foreground = (histogram_norm[t+1:] * np.arange(t+1, 256)).sum() / max(prob_foreground, epsilon)

    # Compute intra-class variance
    intra_class_variance = prob_background * prob_foreground * ((mean_background - mean_foreground) ** 2)

    # Update best threshold if necessary
    if intra_class_variance > best_intra_class_variance:
        best_intra_class_variance = intra_class_variance
        best_threshold = t

# Apply the threshold
binary_image = cv2.threshold(image, best_threshold, 255, cv2.THRESH_BINARY)[1]

# Define the erosion kernel (structuring element)
kernel = np.ones((4, 4), dtype=np.uint8)  # 3x3 square kernel

# Perform binary erosion
dilated_image = cv2.dilate(binary_image, kernel, iterations=13)
eroded_image = cv2.erode(dilated_image, kernel, iterations=14)

# Display the original image and the binary image
cv2.imshow('Original Image', image)
cv2.imshow('Binary Image', binary_image)
cv2.imshow('Dilated Image', dilated_image)
cv2.imshow('Eroded Image', eroded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
