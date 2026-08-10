import math

def ndcg(relevance_scores, k):
    """
    Compute NDCG@k.
    """
    DCG = 0
    for idx, ele in enumerate(relevance_scores[:k], 1):
        # print(idx, ele)
        DCG += (2 ** ele - 1) / math.log2(idx + 1)

    IDCG = 0
    for idx, ele in enumerate(sorted(relevance_scores, reverse=True)[:k], 1):
        # print(idx, ele)
        IDCG += (2 ** ele - 1) / math.log2(idx + 1)

    if IDCG == 0:
        return 0

    return DCG / IDCG
