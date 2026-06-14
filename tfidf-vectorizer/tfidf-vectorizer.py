import numpy as np
from collections import Counter
from itertools import product
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    # Write code here
    N = len(documents)
    vocab = {}
    doc_word_freq = {}
    doc_len = []
    for idx, sentence in enumerate(documents):
        words = sentence.split()
        for word in set(words):
            vocab[word] = vocab.get(word, 0) + 1
        doc_word_freq[idx] = Counter(words)
        doc_len.append(len(words))

    tf_idf = np.zeros(shape=(N, len(vocab)))
    for doc_idx, (word_idx, word) in product(range(N), enumerate(sorted(vocab))):
        tf = doc_word_freq[doc_idx].get(word, 0) / doc_len[doc_idx]
        idf = math.log(N / vocab[word])
        tf_idf[doc_idx, word_idx] = tf * idf
    # print(tf_idf)
    return tf_idf, sorted(vocab)
    
