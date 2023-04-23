import cv2
import numpy as np
from FindCellLocations import find_cell_locations

image_list = ['im1.jpg', 'im2.jpg', 'im3.jpg']
mask_image_list = ['im1_gold_mask.txt', 'im2_gold_mask.txt', 'im3_gold_mask.txt']

for img_index in range(len(image_list)):

    # Load the grayscale image
    rgb_image = cv2.imread('data/' + image_list[img_index], cv2.IMREAD_GRAYSCALE)

    # Load the text data into a NumPy array
    mask_image = cv2.imread('output/' + image_list[img_index], cv2.IMREAD_GRAYSCALE)

    # Find the cell locations
    cell_locations = find_cell_locations(rgb_image, mask_image)

    cv2.imshow('image', rgb_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Draw the cell locations
    """
    for cell_location in cell_locations:
        cv2.circle(mask_image, cell_location, 5, (0, 0, 255), -1)

    cv2.imwrite('deneme/' + image_list[img_index], mask_image)
    """

