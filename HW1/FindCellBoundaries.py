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


def region_growing(seeds, image, mask, threshold):
    segmentation_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    seeds.reverse()

    # Create a stack for region growing
    stack = Stack()
    for seed in seeds:
        stack.push(seed)

    # Define 8-connected neighborhood
    neighbors = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    seed_counter = 0

    # Iterate over queue
    while stack.size() > 0:

        # Get the first element from the queue
        x, y = stack.pop()

        if (x, y) in seeds:
            seed = (x, y)
            seed_counter += 1

        if (y < 768 and x < 1024) and mask[y, x] != 0 and segmentation_map[y, x] == 0:
            intensity_difference = abs(int(image[y, x]) - int(image[seed[1], seed[0]]))

            if intensity_difference <= threshold:
                mask[y, x] = 0
                segmentation_map[y, x] = seed_counter

                # Add neighbors of the current pixel to the queue
                for neighbor in neighbors:
                    nx = x + neighbor[0]
                    ny = y + neighbor[1]

                    if (ny < 768 and nx < 1024) and mask[ny, nx] != 0 and segmentation_map[ny, nx] == 0:
                        stack.push((nx, ny))

    return segmentation_map

    # Region growing loop


image_list = ['im1.jpg', 'im2.jpg', 'im3.jpg']
mask_image_list = ['im1_gold_mask.txt', 'im2_gold_mask.txt', 'im3_gold_mask.txt']

for img_index in range(len(image_list)):
    # Load the grayscale image
    rgb_image = cv2.imread('data/' + image_list[img_index], cv2.IMREAD_GRAYSCALE)

    # Load the text data into a NumPy array
    mask_image = cv2.imread('output/' + image_list[img_index], cv2.IMREAD_GRAYSCALE)

    # mask_image[mask_image == 255] = 1

    # Find the cell locations
    cell_locations = find_cell_locations(rgb_image, mask_image)
    segmentation_map = region_growing(cell_locations, rgb_image, mask_image, 20)

    colored = np.zeros((rgb_image.shape[0], rgb_image.shape[1], 3), 'uint8')
    colors = np.random.randint(0, 256, size=(len(cell_locations), 3))
    for i in range(len(cell_locations)):
        colored[segmentation_map == i + 1] = colors[i, :]

    cv2.imwrite('deneme/' + image_list[img_index], colored)

    # Draw the cell locations
    """
    for cell_location in cell_locations:
        cv2.circle(mask_image, cell_location, 5, (0, 0, 255), -1)

    cv2.imwrite('deneme/' + image_list[img_index], mask_image)
    """
