import numpy as np
import copy as cp

from learners.learner import Learner
from preprocess import RecordSet
from typing import Optional


class LinearRegression(Learner):
	"""
	The linear regression learning algorithm. Given data, this class
	constructs and stores a linear mdl that can be applied to
	test data-points to classify them.
	"""

	def __init__(self, alpha: float, **params: any):
		"""
		Initialises the linear regression algorithm.

		:param alpha: Regularization term alpha.
		:param params: Ignored.
		"""
		super().__init__(**params)
		self.name = 'Linear Regression'
		self.alpha = alpha
		self.gamma = 0.5
		self.add_intercept = True
		self.binary_points = True

		self.beta = 0
		self.data: Optional[RecordSet] = None

	def fit(self, rs: RecordSet) -> None:
		"""
		Linear regression using OLS: beta = inv(X'X)X'Y
		"""
		# set params
		self.data = cp.deepcopy(rs)
		patterns = self.data.entries[:, :-1]
		out = self.data.entries[:, -1:]

		if self.add_intercept:
			intercept = np.ones((patterns.shape[0], 1))
			patterns = np.hstack((intercept, patterns))

		self.beta = np.zeros(patterns.shape[1])

		# compute covariance matrix
		identity_matrix = np.identity(patterns.shape[1], dtype=float)
		regularisation = self.alpha * identity_matrix
		covariance_matrix = np.linalg.inv(np.dot(patterns.T, patterns)) + regularisation

		# compute result
		self.beta = np.dot(covariance_matrix, np.dot(patterns.T, out))

	def predict(self, rs: RecordSet) -> np.ndarray:
		"""
		Assigns a predicted class label to the given record sets.

		:param rs: The record set to assign predictions to.
		:return: A column vector of predictions corresponding to the record set's rows.
		"""
		# set params
		patterns = rs.entries[:, :-1]

		if self.add_intercept:
			intercept = np.ones((patterns.shape[0], 1))
			patterns = np.hstack((intercept, patterns))

		# predict
		predictions = np.dot(patterns, self.beta)

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
