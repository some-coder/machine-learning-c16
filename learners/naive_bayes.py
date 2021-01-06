import numpy as np
import copy as cp
from learners.learner import Learner
from preprocess import RecordSet
from typing import Optional
from sklearn.naive_bayes import GaussianNB


class NaiveBayes(Learner):
	"""
	The naive Bayes learning algorithm. Given data, this class
	constructs and stores a relatively efficient Bayesian classifier
	(due to 'naive' assumptions) that can be used to classify test
	data-points.
	"""

	def __init__(self, **params: any) -> None:
		"""
		Constructs a base learning algorithm.

		:param params: Parameter values to supply.
		"""
		super().__init__(**params)
		self.name = 'Naive Bayes Classifier'

		self.data: Optional[RecordSet] = None
		self.bayes_model = None

	def fit(self, rs: RecordSet) -> None:
		"""
		Fits the base learning algorithm to training data.

		:param rs: The record set to provide as training input.
		"""
		self.data = cp.deepcopy(rs)
		x = self.data.entries[:, :-1]
		y = np.ravel(self.data.entries[:, -1:])
		bayes_classifier = GaussianNB()
		bayes_classifier.fit(x, y)
		self.bayes_model = bayes_classifier

	def predict(self, rs: RecordSet) -> np.ndarray:
		"""
		Assigns a predicted class label to the given record sets.

		:param rs: The record set to assign predictions to.
		:return: A column vector of predictions corresponding to the record set's rows.
		"""
		predictions: np.ndarray = np.zeros((rs.entries.shape[0], 1))
		for r in range(rs.entries.shape[0]):
			predictions[[r], :] = self.bayes_model.predict(rs.entries[[r], :-1])
		return predictions
