import numpy as np


def calculateIntensityFeatures(patch, binNumber):
    # Calculate average and standard deviation
    average = np.mean(patch)
    std_dev = np.std(patch)

    # Calculate histogram
    hist, _ = np.histogram(patch, bins=binNumber, range=(0, 255))

    # Calculate probabilities
    total_pixels = patch.size
    probabilities = hist / total_pixels

    # Calculate entropy
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))

    # Return intensity features
    return np.array([average, std_dev, entropy])

