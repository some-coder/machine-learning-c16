import numpy as np
import copy as cp

from sklearn.linear_model import LogisticRegression as LogReg  # avoid naming conflicts

from learners.learner import Learner
from preprocess import RecordSet
from typing import Optional


class LogisticRegression(Learner):
	"""
	The logistic regression learning algorithm. Given data, this class
	constructs and stores a linear mdl with a logistic link function,
	that can be used to classify test data-points.
	"""

	def __init__(self, alpha: float, **params: any):
		"""
		Initialises the Logistic regression algorithm.

		:param alpha: regularization term alpha.
		:param params: Ignored.
		"""
		super().__init__(**params)
		self.name = 'Logistic Regression'
		self.alpha = alpha
		self.gamma = 0.5
		self.binary_points = True

		self.beta = list()
		self.data: Optional[RecordSet] = None
		self.model: Optional[LogReg] = None  # will be built during fit

	def fit(self, rs: RecordSet) -> None:
		"""
		fit a Logistic regression mdl

		:param rs: The record set to fit with.
		"""
		# set params
		self.data = cp.deepcopy(rs)
		patterns = self.data.entries[:, :-1]
		out = self.data.entries[:, -1:]

		# avoid error
		if self.alpha == 0:
			raise Exception("Alpha too low to obtain reliable results")

		# import the logistic regression
		self.model = LogReg(C=1/self.alpha, penalty="l1", solver="liblinear")
		self.model.fit(X=patterns, y=out.ravel())

	def predict(self, rs: RecordSet) -> np.ndarray:
		"""
		Assigns a predicted class label to the given record sets.

		:param rs: The record set to assign predictions to.
		:return: A column vector of predictions corresponding to the record set's rows.
		"""
		# set params
		patterns = rs.entries[:, :-1]

		# predict
		predictions = self.model.predict_proba(X=patterns)[:, 1]

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

	@staticmethod
	def logistic_function(x: float) -> float:
		"""
		Computes the output of applying the value to the logistic function.

		:param x: The value to evaluate under the logistic function.
		:return: The outcome.
		"""
		return 1 / (1 + np.exp(-x))
