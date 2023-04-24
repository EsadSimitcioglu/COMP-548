import cv2
import numpy as np
from FindCellLocations import find_cell_locations

from metric import intersection_over_union, dice_index


class Stack:
    def __init__(self):
        """
        Initialize an empty stack.
        """
        self.items = []

    def push(self, item):
        """
        Add an item to the top of the stack.

        Args:
        - item: Any, item to be added to the stack
        """
        self.items.append(item)

    def pop(self):
        """
        Remove and return the item from the top of the stack.

        Returns:
        - item: Any, item removed from the top of the stack
        """
        if not self.is_empty():
            return self.items.pop()

    def peek(self):
        """
        Return the item from the top of the stack without removing it.

        Returns:
        - item: Any, item from the top of the stack
        """
        if not self.is_empty():
            return self.items[-1]

    def is_empty(self):
        """
        Check if the stack is empty.

        Returns:
        - bool: True if the stack is empty, False otherwise
        """
        return len(self.items) == 0

    def size(self):
        """
        Return the number of items in the stack.

        Returns:
        - int: Number of items in the stack
        """
        return len(self.items)


def region_growing(seed, image, mask, threshold, segmentation_map, seed_counter):
    # Create a stack for region growing
    stack = Stack()

    stack.push(seed)

    # Define 8-connected neighborhood
    neighbors = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    best_threshold = 185

    binary_image = cv2.threshold(rgb_image, best_threshold, 255, cv2.THRESH_BINARY)[1]
    opened_binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1)
    cells = cv2.bitwise_not(opened_binary_image, mask=mask_image)

    # do edge canny detector
    eroded_cells = cv2.erode(cells, kernel, iterations=1)
    dilated_cells = cv2.dilate(eroded_cells, kernel, iterations=1)

    # Apply median filter to remove noise
    ksize = 5  # Kernel size for median filter
    median_filter_image = cv2.medianBlur(dilated_cells, ksize)

    # do dilation
    dilated_image = cv2.morphologyEx(median_filter_image, cv2.MORPH_CLOSE, kernel, iterations=1)

    # do distance transform
    dist = cv2.distanceTransform(dilated_image, cv2.DIST_L2, 3)

    # Display the distance transform
    # cv2.imshow('Distance Transform', dist)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Iterate over queue
    while stack.size() > 0:

        # Get the first element from the queue
        x, y = stack.pop()
        if (y < 768 and x < 1024) and mask[y, x] != 0 and dist[y, x] != 0 and segmentation_map[y, x] == 0:
            intensity_difference = abs(int(image[y, x]) - int(image[seed[1], seed[0]]))

            if intensity_difference <= threshold:
                mask[y, x] = 0
                segmentation_map[y, x] = seed_counter

                # Add neighbors of the current pixel to the queue
                for neighbor in neighbors:
                    nx = x + neighbor[0]
                    ny = y + neighbor[1]

                    if (ny < 768 and nx < 1024) and mask[ny, nx] != 0 and dist[y, x] != 0 and segmentation_map[
                        ny, nx] == 0:
                        stack.push((nx, ny))

    return segmentation_map


image_list = ['im1.jpg', 'im2.jpg', 'im3.jpg']
mask_image_list = ['im1_gold_mask.txt', 'im2_gold_mask.txt', 'im3_gold_mask.txt']
ground_truth_list = ['im1_gold_cells.txt', 'im2_gold_cells.txt', 'im3_gold_cells.txt']
threshold_list = [0.5, 0.75, 0.9]

for img_index in range(len(image_list)):
    # Load the grayscale image
    rgb_image = cv2.imread('data/' + image_list[img_index], cv2.IMREAD_GRAYSCALE)

    # Load the text data into a NumPy array
    mask_image = cv2.imread('mask/' + image_list[img_index], cv2.IMREAD_GRAYSCALE)

    # Load the text data into a NumPy array
    ground_truth = np.loadtxt('data/' + ground_truth_list[img_index], dtype=np.uint32)

    # Find the cell locations
    cell_locations = find_cell_locations(rgb_image, mask_image, img_index)

    segmentation_map = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.int32)
    seed_counter = 1
    for cell in cell_locations:
        segmentation_map = region_growing(cell, rgb_image, mask_image, 50, segmentation_map, seed_counter)
        seed_counter += 1

    colored = np.zeros((rgb_image.shape[0], rgb_image.shape[1], 3), 'uint8')
    colors = np.random.randint(0, 256, size=(len(cell_locations), 3))
    for i in range(len(cell_locations)):
        colored[segmentation_map == i + 1] = colors[i, :]

    cv2.imwrite('colored_cell/' + image_list[img_index], colored)

    """
                                                            
    for threshold in threshold_list:
        precision, recall, f_score = intersection_over_union(segmentation_map, ground_truth, threshold)
        print("***********************")
        print("Precision:" + str(precision))
        print("Recall: " + str(recall))
        print("F-Score: " + str(f_score))
        print("***********************")
        """

    dice_metric = dice_index(segmentation_map, ground_truth)
    print("Dice Metric: " + str(dice_metric))
