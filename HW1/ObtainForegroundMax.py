import cv2
import numpy as np

from metric import pixel_level

img_list = ['im1.jpg', 'im2.jpg', 'im3.jpg']
ground_truth_list = ['im1_gold_mask.txt', 'im2_gold_mask.txt', 'im3_gold_mask.txt']

for img_index in range(len(img_list)):

    # Load the grayscale image
    test_image = cv2.imread('data/' + img_list[img_index], cv2.IMREAD_GRAYSCALE)

    # Load the text data into a NumPy array
    ground_truth = np.loadtxt('data/' + ground_truth_list[img_index])

    # Calculate histogram
    histogram = cv2.calcHist([test_image], [0], None, [256], [0, 256])

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
        mean_background = (histogram_norm[:t + 1] * np.arange(t + 1)).sum() / max(prob_background, epsilon)
        mean_foreground = (histogram_norm[t + 1:] * np.arange(t + 1, 256)).sum() / max(prob_foreground, epsilon)

        # Compute intra-class variance
        intra_class_variance = prob_background * prob_foreground * ((mean_background - mean_foreground) ** 2)

        # Update best threshold if necessary
        if intra_class_variance > best_intra_class_variance:
            best_intra_class_variance = intra_class_variance
            best_threshold = t

    edges = cv2.Canny(test_image, 100, 200)  # You can adjust the thresholds as needed

    binary_image = cv2.threshold(test_image, best_threshold, 255, cv2.THRESH_BINARY)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # dilated_image = cv2.dilate(binary_image, kernel, iterations=20)
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=20)

    cv2.imwrite('mask/' + img_list[img_index], closed_image)

    # cv2.imshow('Dilated Image', closed_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    closed_image[closed_image == 255] = 1

    precision, recall, f_score = pixel_level(ground_truth, closed_image)

    print("Precision:" + str(precision))
    print("Recall: " + str(recall))
    print("F-Score: " + str(f_score))

    print("***********************")
