import cv2
import numpy as np

from metric import pixel_level

ground_truth = np.loadtxt('data/im1_gold_mask.txt')
rgb_image = cv2.imread('data/im1.jpg', cv2.IMREAD_GRAYSCALE)

pixel_gray_value_list = list()

for row in range(768):
    for column in range(1024):

        if ground_truth[row][column] == 1 and rgb_image[row][column] > 180:
            ground_truth[row][column] = 0
            pixel_gray_value_list.append(rgb_image[row][column])


# Calculate dictionary of pixel gray values and their frequency
pixel_gray_value_dict = dict()
for pixel_gray_value in pixel_gray_value_list:
    if pixel_gray_value in pixel_gray_value_dict:
        pixel_gray_value_dict[pixel_gray_value] += 1
    else:
        pixel_gray_value_dict[pixel_gray_value] = 1

print(pixel_gray_value_dict)

cv2.imshow('Original Image', ground_truth)
cv2.waitKey(0)
cv2.destroyAllWindows()
