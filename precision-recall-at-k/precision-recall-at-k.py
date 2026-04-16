def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    top_k = set(recommended[:k])
    precison_k = len(top_k.intersection(relevant)) / k
    recall_k = len(top_k.intersection(relevant)) / len(relevant)
    return [precison_k, recall_k]