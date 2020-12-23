import numpy as np
import copy as cp

from sklearn.linear_model import LogisticRegression

from learners.learner import Learner
from preprocess import RecordSet
from typing import Optional


class Logistic_Reggression(Learner):
    def __init__(self, alpha: float, **params: any):
        """
        Initialises the Logistic regression algorithm.

		:param alpha: regularization term alpha.
		:param params: Ignored.
        """
        super().__init__(**params)
        self.alpha = alpha
        self.gamma = 0.5
        self.binary_points = True

        self.beta = list()
        self.data: Optional[RecordSet] = None
        return

    def fit(self, rs: RecordSet) -> None:
        """
        fit a Logistic regression model

        :param rs: The record set to fit with.
        """
        # set params
        self.data = cp.deepcopy(rs)
        X = self.data.entries[:, :-1]
        Y = self.data.entries[:, -1:]

        # avoid error
        if self.alpha == 0:
            raise Exception("Alpha too low to obtain reliable results")

        # import the logistic regression
        self.model = LogisticRegression(C=1/self.alpha, penalty="l1", solver="liblinear")
        self.model.fit(X=X, y=Y.ravel())
        return


    def predict(self, rs: RecordSet) -> np.ndarray:
        """
         Assigns a predicted class label to the given record sets.

         :param rs: The record set to assign predictions to.
         :return: A column vector of predictions corresponding to the record set's rows.
         """
        # set params
        X = rs.entries[:, :-1]

        # predict
        predictions = self.model.predict_proba(X=X)[:,1]

        if self.binary_points:
            predictions = self.discrete_points(predictions=predictions)

        # return 2d
        predictions = np.reshape(predictions, (-1, 1))
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

    def logistic_function(self, x):
        return 1 / (1 + np.exp(-x))