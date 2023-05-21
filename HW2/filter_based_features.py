import cv2
import numpy as np


def create_gabor_filters(num_filters=16, ksize=35, sigma=3.0, lambd=10.0, gamma=0.5, psi=0):
    filters = []
    for theta in np.arange(0, np.pi, np.pi / num_filters):
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
        kern /= 1.0 * kern.sum()
        filters.append(kern)
    return filters


def apply_filters(img, filters):
    new_image = np.zeros_like(img)
    for kern in filters:
        image_filter = cv2.filter2D(img, -1, kern)
        np.maximum(new_image, image_filter, new_image)
    return new_image