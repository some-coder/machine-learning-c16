import numpy as np
import copy as cp

from learners.learner import Learner
from preprocess import RecordSet
from typing import Optional

class Linear_Reggression(Learner):
    def __init__(self, alpha: float, **params: any):
        """
        Initialises the linear regression algorithm.

		:param alpha: regularization term alpha.
		:param params: Ignored.
        """
        super().__init__(**params)
        self.alpha = alpha
        self.gamma = 0.5
        self.add_intercept = True
        self.binary_points = True

        self.beta = 0
        self.data: Optional[RecordSet] = None
        return

    def fit(self, rs: RecordSet) -> None:
        """
        Linear regression using OLS: beta = inv(X'X)X'Y
        """
        # set params
        self.data = cp.deepcopy(rs)
        X = self.data.entries[:, :-1]
        Y = self.data.entries[:, -1:]

        if self.add_intercept:
            intercept = np.ones((X.shape[0], 1))
            X = np.hstack((intercept, X))

        self.beta = np.zeros(X.shape[1])

        # compute covariance matrix
        identity_matrix = np.identity(X.shape[1], dtype = float)
        regualization = self.alpha * identity_matrix
        covariance_matrix = np.linalg.inv(np.dot(X.T, X)) + regualization

        # compute result
        self.beta = np.dot(covariance_matrix, np.dot(X.T, Y))
        return

    def predict(self, rs: RecordSet) -> np.ndarray:
        """
        Assigns a predicted class label to the given record sets.

        :param rs: The record set to assign predictions to.
        :return: A column vector of predictions corresponding to the record set's rows.
        """
        # set params
        X = rs.entries[:, :-1]

        if self.add_intercept:
            intercept = np.ones((X.shape[0], 1))
            X = np.hstack((intercept, X))

        # predict
        predictions = np.dot(X, self.beta)

        if self.binary_points:
            predictions = self.discrete_points(predictions=predictions)
        return predictions

    def discrete_points(self, predictions):
        """
        Turns probabilities into discrete classes

        :param predictions: The predicted class probabilities
        :return: A vector with discrete classes
        """
        n = predictions.shape[0]
        for i in range(0, n):
            if predictions[i] >= self.gamma:
                predictions[i] = 1
            else:
                predictions[i] = 0
        return predictions

