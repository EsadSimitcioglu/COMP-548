import numpy as np
import cv2
from skimage.filters import gabor_kernel


def calculateGaborFilter(patch):

    # Initialize the feature vector
    feature_vector = []

    # Define Gabor filter parameters
    frequency = 0.6
    theta = 0.8
    sigma = 5

    # Create the Gabor kernel
    kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma))

    # Apply the Gabor kernel to the patch
    filtered_patch = np.abs(cv2.filter2D(patch, cv2.CV_64F, kernel))

    # Calculate the average magnitude of Gabor responses for the patch
    feature = np.mean(filtered_patch)

    # Append the feature to the feature vector
    feature_vector.append(feature)

    return np.array(feature_vector)
