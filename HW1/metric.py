
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
            elif ground_truth_value == 1 and test_value == 0:
                fp += 1
            elif ground_truth_value == 0 and test_value == 1:
                fn += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * ((precision * recall) / (precision + recall))

    return precision, recall, f_score