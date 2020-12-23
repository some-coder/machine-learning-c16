import numpy as np
import copy as cp

from statsmodels.discrete.discrete_model import Probit

from learners.learner import Learner
from preprocess import RecordSet
from typing import Optional


class Probit_Reggression(Learner):
    def __init__(self, alpha: float, **params: any):
        """
        Initialises the Probit regression algorithm.

		:param alpha: regularization term alpha.
		:param params: Ignored.
        """
        super().__init__(**params)
        self.alpha = alpha
        self.gamma = 0.5
        self.add_intercept = True
        self.binary_points = True

        self.beta = list()
        self.data: Optional[RecordSet] = None
        return

    def fit(self, rs: RecordSet) -> None:
        """
        fit a Probit regression model

        :param rs: The record set to fit with.
        """
        # set params
        self.data = cp.deepcopy(rs)
        X = self.data.entries[:, :-1]
        Y = self.data.entries[:, -1:]

        if self.add_intercept:
            intercept = np.ones((X.shape[0], 1))
            X = np.hstack((intercept, X))

        # avoid error
        if self.alpha == 0:
            raise Exception("Alpha too low to obtain reliable results")

        self.model = Probit(endog=Y.ravel(), exog=X)
        self.model = self.model.fit_regularized(alpha=self.alpha, maxiter=10e5, disp=False)
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
        predictions = self.model.predict(exog=X)

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