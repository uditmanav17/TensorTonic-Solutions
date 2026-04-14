import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)
    padded_seq = []
    for seq in seqs:
        if len(seq) < max_len:
            seq = seq + [pad_value] * (max_len - len(seq))
        elif len(seq) > max_len:
            seq = seq[:max_len]
        padded_seq.append(seq)
    return np.array(padded_seq)