import numpy as np
from collections import Counter

def tfidf_vectorizer(documents):
    tokenized = [doc.split() for doc in documents]

    # Vocabulary
    vocab = sorted(set(word for doc in tokenized for word in doc))
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    N = len(documents)
    V = len(vocab)

    # Term-frequency matrix
    tf = np.zeros((N, V))

    for doc_idx, words in enumerate(tokenized):
        counts = Counter(words)
        doc_len = len(words)

        indices = [word_to_idx[w] for w in counts]
        tf[doc_idx, indices] = (
            np.fromiter(counts.values(), dtype=float) / doc_len
        )

    # Document frequency
    df = (tf > 0).sum(axis=0)

    # IDF
    idf = np.log(N / df)

    # TF-IDF
    tfidf = tf * idf

    return tfidf, vocab