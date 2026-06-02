import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Your code here
    chars, counts = np.unique(tokens, return_counts=True)
    mapping = {char: count for char, count in zip(chars, counts)}
    ans = np.array([mapping.get(i, 0) for i in vocab], dtype=int)
    return ans