import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    u = np.sum(np.exp(x))
    return np.exp(x) / u
