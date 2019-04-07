import numpy as np

class VotingClassifier():
    def __init__(self, estimators=None):
        if estimators is None:
            self.estimators = []
        else:
            self.estimators = estimators

    def add_estimator(self, estimator):
        self.estimators.append(estimator)

    def predict(self, X):
        # get values
        Y = np.zeros([X.shape[0], len(self.estimators)], dtype=int)
        for i, clf in enumerate(self.estimators):
            Y[:, i] = clf.predict(X)
        # apply voting
        y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y[i] = np.argmax(np.bincount(Y[i,:]))
        return y
