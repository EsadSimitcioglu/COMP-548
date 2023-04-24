import cv2
import numpy as np
from metric import pixel_level


img_list = ['d4_h.jpg', 'd7_dr.jpg', 'd11_g.jpg']
ground_truth_list = ['d4_h_gold.png', 'd7_dr_gold.png', 'd11_g_gold.png']

for img_index in range(len(img_list)):

    # Load the grayscale image
    img = cv2.imread('fundus/' + img_list[img_index])

    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply the LoG filter
    ksize = 5  # Kernel size of the LoG filter
    sigma = 1.0  # Standard deviation of the Gaussian kernel
    filtered_image = cv2.GaussianBlur(img_gray, (ksize, ksize), sigma)
    filtered_image = cv2.Laplacian(filtered_image, cv2.CV_64F)

    # Convert the filtered image to the range [0, 255]
    filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)
    filtered_image = filtered_image.astype('uint8')

    # Load the text data into a NumPy array
    ground_truth = cv2.imread('fundus/' + ground_truth_list[img_index], cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur to the image
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Apply the LoG filter
    img_log = cv2.Laplacian(img_blur, cv2.CV_64F, ksize=5)

    # Normalize the output to the range [0, 255]
    img_log = cv2.normalize(img_log, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the output to uint8 data type
    img_log = img_log.astype('uint8')

    # Apply threshold to convert the image into binary
    ret, thresh = cv2.threshold(img_log, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # find contours and discard contours with small areas
    mask = np.zeros_like(thresh)
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    area_thresh = 500
    for cntr in contours:
        area = cv2.contourArea(cntr)
        if area > area_thresh:
            cv2.drawContours(mask, [cntr], -1, 255, 2)

    # apply mask to thresh
    result1 = cv2.bitwise_and(thresh, mask)
    mask = cv2.merge([mask, mask, mask])
    result2 = cv2.bitwise_and(img, mask)

    # convert result2 image to binary image
    result2 = cv2.cvtColor(result2, cv2.COLOR_BGR2GRAY)

    for i in range(result2.shape[0]):
        for j in range(result2.shape[1]):
            if result2[i, j] > 10:
                result2[i, j] = 255
            else:
                result2[i, j] = 0

    # apply dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dist = cv2.distanceTransform(result2, cv2.DIST_L2, 3)
    cv2.normalize(dist, dist, 0, 255, cv2.NORM_MINMAX)

    # Apply median filter to remove noise
    ksize = 5  # Kernel size for median filter
    dist = cv2.medianBlur(dist, ksize)

    # do erosion
    dist = cv2.erode(dist, kernel, iterations=1)

    cv2.imshow('Dilated Image', dist)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    result2[result2 == 255] = 1

    precision, recall, f_score = pixel_level(ground_truth, result2)

    print("Precision:" + str(precision))
    print("Recall: " + str(recall))
    print("F-Score: " + str(f_score))

    print("***********************")
