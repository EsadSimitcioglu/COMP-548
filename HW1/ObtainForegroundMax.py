import cv2
import numpy as np

from HW1.metric import pixel_level

# List of input image filenames
img_list = ['im1.jpg', 'im2.jpg', 'im3.jpg']

# List of ground truth mask filenames
ground_truth_list = ['im1_gold_mask.txt', 'im2_gold_mask.txt', 'im3_gold_mask.txt']

# Loop through each image
for img_index in range(len(img_list)):

    # Load the grayscale image
    test_image = cv2.imread('data/' + img_list[img_index], cv2.IMREAD_GRAYSCALE)

    # Load the ground truth mask data into a NumPy array
    ground_truth = np.loadtxt('data/' + ground_truth_list[img_index])

    # Calculate histogram
    histogram = cv2.calcHist([test_image], [0], None, [256], [0, 256])

    # Normalize the histogram
    histogram_norm = histogram.ravel() / histogram.sum()

    # Compute the cumulative distribution function (CDF)
    cdf = histogram_norm.cumsum()

    # Initialize variables for best threshold and intra-class variance
    best_threshold = 0
    best_intra_class_variance = 0

    # Iterate over all possible threshold values (0-255)
    for t in range(256):
        # Background and foreground class probabilities
        prob_background = cdf[t]
        prob_foreground = 1 - prob_background

        # Background and foreground class means
        epsilon = 1e-8  # small epsilon value to avoid division by zero
        mean_background = (histogram_norm[:t + 1] * np.arange(t + 1)).sum() / max(prob_background, epsilon)
        mean_foreground = (histogram_norm[t + 1:] * np.arange(t + 1, 256)).sum() / max(prob_foreground, epsilon)

        # Compute intra-class variance
        intra_class_variance = prob_background * prob_foreground * ((mean_background - mean_foreground) ** 2)

        # Update best threshold if necessary
        if intra_class_variance > best_intra_class_variance:
            best_intra_class_variance = intra_class_variance
            best_threshold = t

    # Threshold the image using the best threshold value
    binary_image = cv2.threshold(test_image, best_threshold, 255, cv2.THRESH_BINARY)[1]

    # Define a structuring element for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Apply morphological closing operation to fill gaps in the binary image
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=20)

    # Save the resulting closed image as the mask
    cv2.imwrite('mask/' + img_list[img_index], closed_image)

    # Convert the closed image to a binary mask (0 or 1)
    closed_image[closed_image == 255] = 1

    # Compute precision, recall, and F-score using the ground truth mask
    precision, recall, f_score = pixel_level(ground_truth, closed_image)

    # Print the evaluation metrics
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F-Score: " + str(f_score))

    print("***********************")