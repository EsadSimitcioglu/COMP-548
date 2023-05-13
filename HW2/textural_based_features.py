import numpy as np


def calculateCooccurrenceMatrix(patch, binNumber, di, dj):
    # Get the shape of the patch
    patch_shape = patch.shape

    # Create an empty co-occurrence matrix
    M = np.zeros((binNumber, binNumber))

    # Calculate the maximum gray-level value in the patch
    max_gray_level = np.max(patch) + 1

    # Calculate the size of each bin
    bin_size = max_gray_level / binNumber

    # Iterate over each pixel in the patch
    for i in range(patch_shape[0]):
        for j in range(patch_shape[1]):
            # Calculate the current gray-level value
            current_gray_level = patch[i, j]

            # Calculate the bin index for the current pixel
            bin_index = int(current_gray_level // bin_size)

            # Calculate the neighboring pixel coordinates
            neighbor_i = i + di
            neighbor_j = j + dj

            # Check if the neighboring pixel is within the patch boundaries
            if neighbor_i >= 0 and neighbor_i < patch_shape[0] and neighbor_j >= 0 and neighbor_j < patch_shape[1]:
                # Calculate the neighboring gray-level value
                neighbor_gray_level = patch[neighbor_i, neighbor_j]

                # Calculate the bin index for the neighboring pixel
                neighbor_bin_index = int(neighbor_gray_level // bin_size)

                # Increment the co-occurrence count in the matrix
                M[bin_index, neighbor_bin_index] += 1

    return M


def calculateAccumulatedCooccurrenceMatrix(patch, binNumber, d):

    # Given Distance List
    distance_list = [(d, 0), (d, d), (0, d), (-d, d), (-d, 0), (-d, -d), (0, -d), (d, -d)]

    sum_of_calculated_matrices = np.array([[0] * binNumber] * binNumber)

    for distance in distance_list:
        M = calculateCooccurrenceMatrix(patch, binNumber, distance[0], distance[1])
        sum_of_calculated_matrices = np.add(sum_of_calculated_matrices, M)

    return sum_of_calculated_matrices


def calculateCooccurrenceFeatures(accM):
    # Normalized accumulated co-occurrence matrix
    accM = accM / np.sum(accM)

    # Angular Second Moment
    asm = np.sum(accM ** 2)

    # Maximum probability
    max_prob = np.max(accM)

    # Inverse Difference Moment
    idm = np.sum(accM / (1 + np.arange(accM.shape[0])) ** 2)

    # Entropy
    entropy = -np.sum(accM * np.log2(accM + 1e-10))

    return np.array([asm, max_prob, idm, entropy])