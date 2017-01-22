import numpy as np
import NMF as nmf

class RandomPredictor:
    def __init__(self, minVal, maxVal):
        self.minv = minVal
        self.maxv = maxVal

    def fit(self, data):
        (n, m) = np.shape(data)
        self.vecLen = m
        self.numOfUsers = n

    def predict(self, number):
        if(number < self.numOfUsers):
            return np.floor(np.random.uniform(self.minv, self.maxv+1, size=self.vecLen)).astype(int)
        else:
            raise ValueError('No item with number ' + str(number))


class AveragePredictor:
    def __init__(self, b):
        self.b = b

    def fit(self, data):
        self.avg = data.mean()
        self.data = data

    def predict(self, number):
        ratingSum = np.sum(self.data, 0)
        numberOfRatings = np.sum(self.data!=0, 0)
        return (ratingSum + self.b * self.avg)/(numberOfRatings+self.b)

class ViewsPredictor:
    def fit(self, data):
        self.data = data

    def predict(self, number):
        return np.sum(self.data!=0, 0)

class DeviationPredictor:
    def fit(self, data):
        self.data = data

    def predict(self, number):
        v = np.ma.MaskedArray(self.data, self.data<1).std(axis=0)
        for i in range(0, len(v)):
            if str(v[i]) == "--":
                v[i] = -1
        return v


class UserBasedPredictor:
    def __init__(self, K=0, threshold=0):
        self.K = K
        self.threshold = threshold

    def fit(self, data):
        self.data = data
        self.avgM = []
        self.avgData = np.copy(data)
        (usersNum, moviesNum) = np.shape(self.data)
        for movie in range(0, moviesNum):
            vec = self.data[:, movie]
            vec = vec[vec > 0]
            self.avgM.append(np.mean(vec))
            self.avgData[:, movie] = self.avgData[:, movie] - np.mean(vec)
			
		
        self.avgData[self.data == 0] = 0

    def predict(self, number):
        (usersNum, moviesNum) = np.shape(self.data)
        ls = []
        for u in range(0, usersNum):
            fac = self.similarity(number, u)
            if (fac > self.threshold and number != u):
                ls.append(fac)
            else:
                ls.append(0)
        ls = np.array(ls)

        if(self.K>0):
            kth = ls.copy()
            kth.sort()
            kth = kth[-self.K]
            ls[ls<kth] = 0

        moviesls = []
        for m in range(0, moviesNum):
            v = self.avgM[m] + np.sum(self.avgData[:, m] * ls) / np.sum(ls)
            if np.isnan(v):
                moviesls.append(0)
            else:
                moviesls.append(v)
        return np.array(moviesls)

    def similarity(self, u1, u2):
        u1v = self.data[u1]
        u2v = self.data[u2]
        selector = (u1v > 0) & (u2v > 0)
        u1v = u1v[selector] - np.mean(u1v[u1v > 0])
        u2v = u2v[selector] - np.mean(u2v[u2v > 0])
        cor = np.sum(u1v * u2v)
        var1 = np.sqrt(np.sum(u1v * u1v))
        var2 = np.sqrt(np.sum(u2v * u2v))
        return cor / ((var1 * var2) + self.K)


class ItemBasedPredictor:
    def __init__(self, K=0, threshold=0):
        self.K = K
        self.threshold = threshold

    def fit(self, data):
        self.data = data
        self.avgData = np.copy(data)
        (usersNum, moviesNum) = np.shape(self.data)
        for movie in range(0, moviesNum):
            vec = self.data[:, movie]
            vec = vec[vec > 0]
            if len(vec)==0:
                self.avgData[:, movie] = 0
            else:
                self.avgData[:, movie] = self.avgData[:, movie] - np.mean(vec)
        self.avgData[self.data == 0] = 0

    def predict(self, number):
        (usersNum, moviesNum) = np.shape(self.data)

        usermean = self.data[number]
        usermean = np.mean(usermean[usermean > 0])

        rat = [];
        for m in range(0, moviesNum):
            ls = []
            for sm in range(0, moviesNum):
                sim = self.similarity(sm, m)
                if m != sm and sim > self.threshold:
                    ls.append(sim)
                else:
                    ls.append(0)
            ls = np.array(ls)

            if (self.K > 0):
                kth = ls.copy()
                kth.sort()
                kth = kth[-self.K]
                ls[ls < kth] = 0
            v = (np.sum(self.avgData[number] * ls) / (np.sum(ls)) + usermean)
            if np.isnan(v):
                rat.append(0)
            else:
                rat.append(v)
        return np.array(rat)

    def similarity(self, u1, u2):
        u1v = self.data[:, u1]
        u2v = self.data[:, u2]

        selector = (u1v > 0) & (u2v > 0)
        u1v = u1v[selector] - np.mean(u1v[u1v > 0])
        u2v = u2v[selector] - np.mean(u2v[u2v > 0])
        cor = np.sum(u1v * u2v)
        var1 = np.sqrt(np.sum(u1v * u1v))
        var2 = np.sqrt(np.sum(u2v * u2v))
        return cor / ((var1 * var2) + self.K)


class SlopeOnePredictor:
    def fit(self, data):
        self.data = data

    def predict(self, number):
        (usersNum, moviesNum) = np.shape(self.data)

        retscores = []
        for movieId in range(0, moviesNum):
            scores = []
            weight = []
            for m in range(0, moviesNum):
                if m != movieId:
                    s, w = self.dev(m, movieId)
                    if s == None:
                        s,w = 0,0
                    scores.append(s)
                    weight.append(w)
                else:
                    scores.append(0)
                    weight.append(0)

            pred = np.sum((self.data[number] - np.array(scores)) * np.array(weight)) / np.sum(np.array(weight))
            if np.isnan(pred):
                retscores.append(0)
            else:
                retscores.append(pred)
        return np.array(retscores)

    def dev(self, m1, m2):
        v1 = self.data[:, m1]
        v2 = self.data[:, m2]

        selector = (v2 > 0) & (v1 > 0)

        v1 = v1[selector]
        v2 = v2[selector]

        if len(v1)==0:
            return (0, 0)
        return (np.sum(v1 - v2) / len(v1), len(v1))

class NMFPredictor:

    def __init__(self, rank=10, max_iter=100, eta=0.01):
        self.rank = rank
        self.max_iter = max_iter
        self.eta = eta

    def fit(self, data):
        self.data = data
        self.predictor = nmf.NMF(self.rank, self.max_iter, self.eta)
        self.predictor.fit(self.data)
        self.matrix = self.predictor.predict_all()

    def predict(self, number):
        return self.matrix[number,:]