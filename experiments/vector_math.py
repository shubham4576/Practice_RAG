import numpy as np


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


def euclidean_distance(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.linalg.norm(a - b)


def dot_product(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b)