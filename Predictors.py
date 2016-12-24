import numpy as np

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
        return np.ma.MaskedArray(self.data, self.data<1).std(axis=0)
