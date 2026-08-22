import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    predictions = np.asarray(predictions)

    # Process each sample (column)
    result = []

    for sample in predictions.T:
        classes, counts = np.unique(sample, return_counts=True)

        # np.unique sorts classes, so argmax naturally picks
        # the smallest class in case of a tie.
        result.append(classes[np.argmax(counts)])

    return result
