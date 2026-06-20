import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    chars, counts = np.unique(y_train, return_counts=True)
    max_idx = np.argmax(counts)
    # print(chars, counts, max_idx)
    return np.array([chars[max_idx]] * len(X_test))