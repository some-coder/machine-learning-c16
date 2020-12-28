import numpy as np
from learners.learner import Learner
from preprocess import RecordSet
from sklearn import svm


class SVM(Learner):

	def __init__(self, **params: any):
		"""
		Initialises the Support Vector Machine learning algorithm.

		:param params: The parameters to supply to the support vector machine; see sklearn.svm.SVC for details.
		"""
		super().__init__(**params)
		self.classifier: svm.SVC = svm.SVC(**params)

	def fit(self, rs: RecordSet) -> None:
		"""
		Fits the SVC to the provided dataset.

		:param rs: The record set to fit with.
		"""
		x: np.ndarray = rs.entries[:, :-1]
		y: np.ndarray = rs.entries[:, -1]  # 1D array expected, not a 2D column vector
		self.classifier.fit(x, y)

	def predict(self, rs: RecordSet) -> np.ndarray:
		"""
		Assigns a predicted class label to the given record sets.

		:param rs: The record set to assign predictions to.
		:return: A column vector of predictions corresponding to the record set's rows.
		"""
		x: np.ndarray = rs.entries[:, :-1]
		y: np.ndarray = self.classifier.predict(x)  # 1D
		return y.reshape((y.shape[0], 1))
