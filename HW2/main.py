import cv2
import numpy as np
import pandas as pd

from HW2.intensity_based_features import calculateIntensityFeatures

def cropPatch(image, center, patch_size):
    # Calculate the top-left corner of the patch
    x_center, y_center = center
    half_patch_size = patch_size // 2
    x_start = x_center - half_patch_size
    y_start = y_center - half_patch_size

    # Draw a patch in the image
    # patch = cv2.rectangle(image, (x_start, y_start), (x_start + patch_size, y_start + patch_size), (0, 0, 0), 1)

    # Crop the patch from the image
    patch = image[y_start: y_start + patch_size, x_start: x_start + patch_size]

    calculateIntensityFeatures(patch, 10)

    return patch


def displayPatch(patch):
    cv2.imshow('Patch', patch)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread('nucleus-dataset/test_1.png', cv2.IMREAD_GRAYSCALE)

cell_locations = pd.read_csv('nucleus-dataset/test_1_cells', sep='\t', header=None)

for location in cell_locations.values:
    patch = cropPatch(img, (location[0], location[1]), 36)
displayPatch(patch)
