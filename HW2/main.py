import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from HW2.filter_based_features import create_gaborfilter, apply_filter
from intensity_based_features import calculateIntensityFeatures
from textural_based_features import calculateCooccurrenceFeatures, calculateAccumulatedCooccurrenceMatrix
import itertools


def color_cluster(cell_locations, img, cluster_labels):
    # Define colors for each cluster
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (0, 165, 255), (128, 0, 128),
              (255, 255, 0)]  # Add more colors if needed

    # Convert the grayscale image to RGB for visualization
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Visualize each cell with the assigned cluster color
    for i, location in enumerate(cell_locations.values):
        x, y = location[0], location[1]
        cluster_label = cluster_labels[i]
        color = colors[cluster_label]
        cv2.circle(img_rgb, (x, y), 4, color=color, thickness=-1)

    # Show the image with clustered cells
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()


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


def experiment(bin_number, d, N, k):
    cluster_ratio_list = []
    test_nums = [1, 10]
    for test_num in test_nums:

        img = cv2.imread('nucleus-dataset/test_{}.png'.format(test_num), cv2.IMREAD_GRAYSCALE)  # read it in main func
        cell_locations = pd.read_csv('nucleus-dataset/test_{}_cells'.format(test_num), sep='\t', header=None)

        cell_dict = create_cell_dict(cell_locations)

        all_features = np.array([]).reshape(0, 7)

        cell_types = []

        for location in cell_locations.values:
            patch = cropPatch(img, (location[0], location[1]), N)

            # Calculate the intensity-based features
            intensity_feature_vector = calculateIntensityFeatures(patch, bin_number)

            # Calculate the accumulated co-occurrence matrix
            accM = calculateAccumulatedCooccurrenceMatrix(patch, bin_number, d)

            # Calculate the texture-based features
            texture_feature_vector = calculateCooccurrenceFeatures(accM)

            overall_feature_vector = np.concatenate((intensity_feature_vector, texture_feature_vector))

            all_features = np.vstack((all_features, overall_feature_vector))
            cell_types.append(location[2])
        # Normalize the feature vectors
        all_features = (all_features - all_features.min(axis=0)) / (all_features.max(axis=0) - all_features.min(axis=0))

        # Weight the feature vectors
        all_features = weighted_clustering(all_features, cell_dict)

        # Specify the desired number of clusters for k-means

        # Run k-means clustering
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(all_features)

        # Get the cluster labels assigned to each cell
        cluster_labels = kmeans.labels_
        # Get the unique cluster labels
        unique_labels = np.unique(cluster_labels)

        # Create a dictionary to store the cell counts in each cluster
        cell_counts = {label: {'inflammation': 0, 'epithelial': 0, 'spindle': 0} for label in unique_labels}

        # Count the cell types in each cluster
        for i, label in enumerate(cluster_labels):
            cell_counts[label][cell_types[i]] += 1

        # Calculate the ratios of cell types in each cluster
        ratios = {}
        for label, counts in cell_counts.items():
            total_cells = sum(counts.values())
            ratios[label] = {cell_type: count / total_cells for cell_type, count in counts.items()}

        # Print the ratios of cell types in each cluster
        print("Cluster\tinflammation\tepithelial\tspindle")
        accuracy = 0
        for label, ratio in ratios.items():
            max_ratio = max(ratio['inflammation'], ratio['epithelial'], ratio['spindle'])
            accuracy += max_ratio / len(ratios)

            print(f"{label}\t{ratio['inflammation']:.2f}\t\t{ratio['epithelial']:.2f}\t\t{ratio['spindle']:.2f}")

        print('accuracy: ', accuracy)
        # Get the cluster centers
        cluster_centers = kmeans.cluster_centers_

        # Get the number of cells in each cluster
        unique, counts = np.unique(cluster_labels, return_counts=True)
        cluster_sizes = dict(zip(unique, counts))

        # Print the number of cells in each cluster
        print(cluster_sizes)

        color_cluster(cell_locations, img, cluster_labels)

        total_sum = sum(cluster_sizes)
        cluster_ratios = [(num / total_sum) * 100 for num in cluster_sizes]

        cluster_ratio_list.append(cluster_ratios)

    return cluster_ratio_list


is_filter_apply = True

bin_number_vals = [i for i in [5, 10, 15]]
# bin_number_vals = [10]
d_vals = [i for i in [1, 2, 3]]
# d_vals = [3]
N_vals = [i for i in [12, 36, 42]]
# N_vals = [12]
k_vals = [3, 5]
# k_vals = [3]

test_nums = [1, 10]
cell_dict_list = []
for testNum in test_nums:
    if is_filter_apply:
        img = cv2.imread('nucleus-dataset/test_{}.png'.format(testNum), cv2.IMREAD_GRAYSCALE)  # read it in main func
        gfilters = create_gaborfilter()
        img = apply_filter(img, gfilters)
    else:
        img = cv2.imread('nucleus-dataset/test_{}.png'.format(testNum), cv2.IMREAD_GRAYSCALE)  # read it in main func

    cell_locations = pd.read_csv('nucleus-dataset/test_{}_cells'.format(testNum), sep='\t', header=None)
    cell_dict = create_cell_dict(cell_locations)
    cell_dict_list.append(cell_dict)

print("Ground Truth: ", cell_dict_list)

for permutation in itertools.product(bin_number_vals, d_vals, N_vals, k_vals):
    print("Testing parameters (bin_number_vals, d_vals, N_vals, k_vals): ", *permutation)
    experiment(*permutation)
    print("--------------------------------------------------")
