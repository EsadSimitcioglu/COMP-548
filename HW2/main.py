import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from HW2.intensity_based_features import calculateIntensityFeatures
from HW2.textural_based_features import calculateCooccurrenceFeatures, calculateAccumulatedCooccurrenceMatrix


def create_cell_dict(cell_locations):
    class_dict = {}

    for location in cell_locations.values:
        if location[2] in class_dict:
            class_dict[location[2]] += 1
        else:
            class_dict[location[2]] = 1

    return class_dict


def weighted_clustering(all_features, cell_dict):
    weights = []
    for class_label, count in cell_dict.items():
        weight = 1.0 / count  # Inverse of class count as weight
        weights.extend([weight] * count)

    weights = np.array(weights)
    weights /= np.sum(weights)  # Normalize weights to sum to 1

    weighted_dataset = all_features * weights[:, np.newaxis]  # Multiply each sample by its weight

    return weighted_dataset


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

cell_dict = create_cell_dict(cell_locations)

print(cell_dict)

d = 1
bin_number = 10

all_features = np.array([]).reshape(0, 7)

for location in cell_locations.values:
    patch = cropPatch(img, (location[0], location[1]), 36)

    features = []

    # Calculate the intensity-based features
    intensity_feature_vector = calculateIntensityFeatures(patch, bin_number)

    accM = calculateAccumulatedCooccurrenceMatrix(patch, bin_number, d)

    # Calculate the texture-based features
    texture_feature_vector = calculateCooccurrenceFeatures(accM)

    overall_feature_vector = np.concatenate((intensity_feature_vector, texture_feature_vector))

    all_features = np.vstack((all_features, overall_feature_vector))

# Normalize the feature vectors
all_features = (all_features - all_features.min(axis=0)) / (all_features.max(axis=0) - all_features.min(axis=0))

# Weight the feature vectors
all_features = weighted_clustering(all_features, cell_dict)

# Specify the desired number of clusters for k-means
num_clusters = 3

# Run k-means clustering
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(all_features)

# Get the cluster labels assigned to each cell
cluster_labels = kmeans.labels_

# Get the cluster centers
cluster_centers = kmeans.cluster_centers_

# Get the number of cells in each cluster
unique, counts = np.unique(cluster_labels, return_counts=True)
cluster_sizes = dict(zip(unique, counts))

# Print the number of cells in each cluster
print(cluster_sizes)
