import numpy as np


def calculateCooccurrenceMatrix(patch, binNumber, di, dj):
    # Calculate histogram
    hist, _ = np.histogram(patch, bins=binNumber, range=(0, 255))

    # Create co-occurrence matrix
    M = np.zeros((binNumber, binNumber), dtype=np.uint32)

    # Calculate co-occurrence matrix
    height, width = patch.shape
    for y in range(height):
        for x in range(width):
            i = patch[y, x]

            # Calculate neighboring pixel coordinates
            ny = y + di
            nx = x + dj

            # Check if neighboring pixel is within bounds
            if ny >= 0 and ny < height and nx >= 0 and nx < width:
                j = patch[ny, nx]
                M[i, j] += 1

    return M


def calculateAccumulatedCooccurrenceMatrix(patch, binNumber, d):

    # Given Distance List
    distance_list = [(d, 0), (d, d), (0, d), (-d, d), (-d, 0), (-d, -d), (0, -d), (d, -d)]

    sum_of_calculated_matrices = []

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

    return asm, max_prob, idm, entropy
