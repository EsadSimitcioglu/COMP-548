from copy import deepcopy


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

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * ((precision * recall) / (precision + recall))

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

