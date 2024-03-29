from copy import deepcopy

import numpy as np


def pixel_level(ground_truth_image, test_image):
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for row in range(768):
        for column in range(1024):
            ground_truth_value = ground_truth_image[row][column]
            test_value = test_image[row][column]

            if ground_truth_value == 1 and test_value == 1:
                tp += 1
            elif ground_truth_value == 0 and test_value == 0:
                tn += 1
            elif ground_truth_value == 0 and test_value == 1:
                fp += 1
            elif ground_truth_value == 1 and test_value == 0:
                fn += 1

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f_score = 2 * ((precision * recall) / (precision + recall  + 1e-7))

    return precision, recall, f_score


def cell_level(ground_truth, centroidList):
    tp = 0
    fp = 0
    fn = 0

    cellNumList = []
    for x in range(768):
        for y in range(1024):
            if ground_truth[x][y] not in cellNumList and ground_truth[x][y] != 0:
                cellNumList.append(ground_truth[x][y])
    cellNumCopy = deepcopy(cellNumList)

    for y, x in centroidList:
        ground_truth_val = ground_truth[x][y]
        if ground_truth_val in cellNumList:
            tp += 1
            cellNumList.remove(ground_truth_val)
        elif ground_truth_val == 0:
            fp += 1

    cellNumList = deepcopy(cellNumCopy)
    for cellNum in cellNumList:
        l = [c for c in centroidList if ground_truth[c[1]][c[0]] == cellNum]

        if len(l) != 1:
            fn += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * ((precision * recall) / (precision + recall))

    return precision, recall, f_score


def intersection_over_union(segmented_image, ground_truth, threshold):
    tp = 0

    cellNumList = []
    for x in range(768):
        for y in range(1024):
            if ground_truth[x][y] not in cellNumList and ground_truth[x][y] != 0:
                cellNumList.append(ground_truth[x][y])

    for cell_id in cellNumList:

        segmented_image_cell_counter = 0
        row_indices, col_indices = np.where(ground_truth == cell_id)

        cell_id_occurance_dict = {}
        for i in range(len(row_indices)):
            if segmented_image[row_indices[i]][col_indices[i]] != 0:
                if segmented_image[row_indices[i]][col_indices[i]] in cell_id_occurance_dict:
                    cell_id_occurance_dict[segmented_image[row_indices[i]][col_indices[i]]] += 1
                else:
                    cell_id_occurance_dict[segmented_image[row_indices[i]][col_indices[i]]] = 1

        # Sort the dictionary by value
        sorted_cell_id_occurance_dict = sorted(cell_id_occurance_dict.items(), key=lambda x: x[1], reverse=True)

        # Check if dict is empty
        if not sorted_cell_id_occurance_dict:
            continue

        # Get the key and value of the first element in the sorted dictionary
        cell_id_of_segmented_image, occurance_of_cell_id = sorted_cell_id_occurance_dict[0]

        # Iterate over segmented_image array
        for i in range(768):
            for j in range(1024):
                if segmented_image[i][j] == cell_id_of_segmented_image:
                    segmented_image_cell_counter += 1

        area_of_overlap = occurance_of_cell_id
        area_of_union = len(row_indices) + segmented_image_cell_counter - area_of_overlap
        iou = area_of_overlap / area_of_union

        if iou > threshold:
            tp += 1

    number_of_ground_truth_cells = len(cellNumList)
    number_of_segmented_cells = len(np.unique(segmented_image)) - 1

    fn = number_of_ground_truth_cells - tp
    fp = number_of_segmented_cells - tp

    print("tp: ", tp)
    print("fp: ", fp)
    print("fn: ", fn)

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f_score = 2 * ((precision * recall) / (precision + recall + 1e-7))

    return precision, recall, f_score


def maximal_overlap(src, dst, cur):
    maximal, index = 0, 0
    for i in range(1, np.max(src) + 1):
        cur_sum = np.sum(np.logical_and(src == i, dst == cur))
        if maximal < cur_sum:
            maximal, index = cur_sum, i
    DI = 2 * maximal / (np.sum(src == index) + np.sum(dst == cur))
    return index, DI


def dice_index(ground_truth, segmented_image):
    res = 0

    segm_sum = np.sum(segmented_image != 0)
    ground_sum = np.sum(ground_truth != 0)

    for i in range(1, np.max(segmented_image) + 1):
        ground_index, DI = maximal_overlap(ground_truth, segmented_image, i)
        res += DI * (np.sum(segmented_image == i) / segm_sum)

    for i in range(1, np.max(ground_truth) + 1):
        image_index, DI = maximal_overlap(segmented_image, ground_truth, i)
        res += DI * (np.sum(ground_truth == i) / ground_sum)

    return res / 2
