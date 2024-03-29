
import cv2
import numpy as np

from metric import cell_level


def otsu_method(rgb_image):
    # Calculate histogram
    histogram = cv2.calcHist([rgb_image], [0], None, [256], [0, 256])

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

        # Update the best threshold if necessary
        if intra_class_variance > best_intra_class_variance:
            best_intra_class_variance = intra_class_variance
            best_threshold = t

    return best_threshold


def find_cell_locations(rgb_image, mask_image, i):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    best_threshold = otsu_method(rgb_image)

    binary_image = cv2.threshold(rgb_image, best_threshold, 255, cv2.THRESH_BINARY)[1]
    opened_binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1)
    cells = cv2.bitwise_not(opened_binary_image, mask=mask_image)
    cells = cv2.threshold(cells, best_threshold, 255, cv2.THRESH_BINARY)[1]
    eroded_cells = cv2.erode(cells, kernel, iterations=1)

    dist = cv2.distanceTransform(eroded_cells, cv2.DIST_L2, 3)
    cv2.normalize(dist, dist, 0, 255, cv2.NORM_MINMAX)

    # save distance transform image
    cv2.imwrite('distance_transform/im{}.jpg'.format(i), dist)

    # dilation
    dist = cv2.morphologyEx(dist, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Apply median filter to remove noise
    ksize = 5  # Kernel size for median filter
    dist = cv2.medianBlur(dist, ksize)

    # do dilation
    dist = cv2.morphologyEx(dist, cv2.MORPH_CLOSE, kernel, iterations=2)

    binary = dist.astype('uint8')

    # Find contours in the binary image
    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    centroidList = []
    # Iterate through contours and calculate x and y distances
    for contour in contours:
        # Calculate the centroid of the contour
        M = cv2.moments(contour)
        # Skip the contour if the area is zero (i.e., M["m00"] == 0)
        if M["m00"] == 0:
            continue

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centroidList.append((cX, cY))


    # cv2.imshow('Binary', dist)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return centroidList


for i in range(1, 4):
    rgb_image = cv2.imread('data/im{}.jpg'.format(i), cv2.IMREAD_GRAYSCALE)
    mask_image = cv2.imread('mask/im{}.jpg'.format(i), cv2.IMREAD_GRAYSCALE)
    centroidList = find_cell_locations(rgb_image, mask_image, i)
    ground_truth = np.loadtxt('data/im{}_gold_cells.txt'.format(i))
    print(cell_level(ground_truth, centroidList))
