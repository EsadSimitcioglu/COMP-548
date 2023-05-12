import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from HW2.intensity_based_features import calculateIntensityFeatures
from HW2.textural_based_features import calculateCooccurrenceFeatures, calculateAccumulatedCooccurrenceMatrix

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

# Concatenate the feature vectors of all cells across all images
all_features = []

d = 1
bin_number = 10

for location in cell_locations.values:
    patch = cropPatch(img, (location[0], location[1]), 36)

    features = []

    # Calculate the intensity-based features
    features.append(calculateIntensityFeatures(patch, bin_number))

    accM = calculateAccumulatedCooccurrenceMatrix(patch, bin_number, d)

    # Calculate the texture-based features
    features.append(calculateCooccurrenceFeatures(accM))

    # Assuming you have a nested list 'features' that contains the feature vectors for each cell in each image
    for image_features in features:
        all_features.extend(image_features)

    # Convert the concatenated feature vectors to a NumPy array
    all_features = np.array(all_features)

# Specify the desired number of clusters for k-means
num_clusters = 5

# Run k-means clustering
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(all_features)

# Get the cluster labels assigned to each cell
cluster_labels = kmeans.labels_

# Get the cluster centers
cluster_centers = kmeans.cluster_centers_

# Show the cluster centers
for cluster_center in cluster_centers:
    displayPatch(cluster_center.reshape((36, 36)))


