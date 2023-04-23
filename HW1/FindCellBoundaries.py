import cv2
import numpy as np
from FindCellLocations import find_cell_locations


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
    eroded_cells = cv2.erode(cells, kernel, iterations=1)

    # Iterate over queue
    while stack.size() > 0:

        # Get the first element from the queue
        x, y = stack.pop()

        if (y < 768 and x < 1024) and mask[y, x] != 0 and eroded_cells[y, x] != 0 and segmentation_map[y, x] == 0:
            intensity_difference = abs(int(image[y, x]) - int(image[seed[1], seed[0]]))

            if intensity_difference <= threshold:
                mask[y, x] = 0
                segmentation_map[y, x] = seed_counter

                # Add neighbors of the current pixel to the queue
                for neighbor in neighbors:
                    nx = x + neighbor[0]
                    ny = y + neighbor[1]

                    if (ny < 768 and nx < 1024) and mask[ny, nx] != 0 and eroded_cells[y, x] != 0 and segmentation_map[
                        ny, nx] == 0:
                        stack.push((nx, ny))

    return segmentation_map


image_list = ['im1.jpg', 'im2.jpg', 'im3.jpg']
mask_image_list = ['im1_gold_mask.txt', 'im2_gold_mask.txt', 'im3_gold_mask.txt']


for img_index in range(len(image_list)):
    # Load the grayscale image
    rgb_image = cv2.imread('data/' + image_list[img_index], cv2.IMREAD_GRAYSCALE)

    # Load the text data into a NumPy array
    mask_image = cv2.imread('mask/' + image_list[img_index], cv2.IMREAD_GRAYSCALE)

    # Find the cell locations
    cell_locations = find_cell_locations(rgb_image, mask_image)

    segmentation_map = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.uint8)
    seed_counter = 1
    for cell in cell_locations:
        segmentation_map = region_growing(cell, rgb_image, mask_image, 20, segmentation_map, seed_counter)
        seed_counter += 1

    colored = np.zeros((rgb_image.shape[0], rgb_image.shape[1], 3), 'uint8')
    colors = np.random.randint(0, 256, size=(len(cell_locations), 3))
    for i in range(len(cell_locations)):
        colored[segmentation_map == i + 1] = colors[i, :]

    cv2.imwrite('colored_cell/' + image_list[img_index], colored)
