import cv2
import numpy as np
from metric import pixel_level

img_list = ['d4_h.jpg', 'd7_dr.jpg', 'd11_g.jpg']
ground_truth_list = ['d4_h_gold.png', 'd7_dr_gold.png', 'd11_g_gold.png']

for img_index in range(len(img_list)):

    # Load the grayscale image
    img = cv2.imread('fundus/' + img_list[img_index], cv2.IMREAD_GRAYSCALE)

    # Load the text data into a NumPy array
    ground_truth = cv2.imread('fundus/' + ground_truth_list[img_index], cv2.IMREAD_GRAYSCALE)


    # Apply Gaussian blur to the image
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply the LoG filter
    img_log = cv2.Laplacian(img_blur, cv2.CV_64F, ksize=5)

    # Normalize the output to the range [0, 255]
    img_log = cv2.normalize(img_log, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the output to uint8 data type
    img_log = img_log.astype('uint8')

    # Apply threshold to convert the image into binary
    ret, thresh = cv2.threshold(img_log, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Define the kernel for erosion and opening
    kernel = np.ones((3, 3), np.uint8)  # You can adjust the size of the kernel as needed

    # find contours and discard contours with small areas
    mask = np.zeros_like(thresh)
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    area_thresh = 1000
    for cntr in contours:
        area = cv2.contourArea(cntr)
        if area > area_thresh:
            cv2.drawContours(mask, [cntr], -1, 255, 2)

    # apply mask to thresh
    result1 = cv2.bitwise_and(thresh, mask)
    mask = cv2.merge([mask,mask,mask])
    #result2 = cv2.bitwise_and(img, mask)

    # do binary dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close = cv2.morphologyEx(result1, cv2.MORPH_CLOSE, kernel, iterations=5)
    dilate = cv2.dilate(result1, kernel, iterations=3)

    dilate[dilate == 255] = 1

    precision, recall, f_score = pixel_level(ground_truth, dilate)

    print("Precision:" + str(precision))
    print("Recall: " + str(recall))
    print("F-Score: " + str(f_score))

    print("***********************")

    dilate[dilate == 1] = 255
    ground_truth[ground_truth == 1] = 255


    # cv2.imshow('Image', ground_truth)
    # cv2.imshow('Dilated Image', dilate)
    # cv2.imshow('Close', close)
    # cv2.imshow('Img Log', thresh)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
