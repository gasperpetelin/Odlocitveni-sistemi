import numpy as np
import itertools

class NMF:
    def __init__(self, rank=10, max_iter=100, eta=0.01):
        self.rank = rank
        self.max_iter = max_iter
        self.eta = eta

    def fit(self, X):
        m, n = X.shape

        W = np.random.rand(m, self.rank)
        H = np.random.rand(n, self.rank)

        w_vars = list(itertools.product(range(m), range(self.rank)))
        h_vars = list(itertools.product(range(n), range(self.rank)))

        nzcols = dict([(j, X[:, j].nonzero()[0]) for j in range(n)])
        nzrows = dict([(i, X[i, :].nonzero()[0]) for i in range(m)])

        self.error = np.zeros((self.max_iter,))

        for t in range(self.max_iter):
            np.random.shuffle(w_vars)
            np.random.shuffle(h_vars)

            for i, k in w_vars:
                selector = (X[i, :] != 0).astype(int)
                W[i, k] = W[i, k] + self.eta * (X[i, :] - H[:, :].dot(W[i, :])).dot(selector) * W[i, k]

            for j, k in h_vars:
                selector = (X[:, j] != 0).astype(int)
                H[j, k] = H[j, k] + self.eta * (X[:, j] - W[:, :].dot(H[j, :])).dot(selector) * H[j, k]

            self.error[t] = np.linalg.norm((X - W.dot(H.T))[X > 0]) ** 2

        self.W = W
        self.H = H
        return (W, H)

    def predict(self, i, j):
        return self.W[i, :].dot(self.H[j, :])

    def predict_all(self):
        return self.W.dot(self.H.T)