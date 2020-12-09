import copy as cp
import numpy as np
import scipy.stats as st
from learners.learner import Learner
from preprocess import RecordSet
from typing import Optional


class KNN(Learner):

	def __init__(self, k: int, **params: any):
		"""
		Initialises the K-Nearest Neighbours learning algorithm.

		:param k: The number of neighbours to compare with. Best set uneven.
		:param params: Ignored.
		"""
		super().__init__(**params)
		self.k = k
		self.data: Optional[RecordSet] = None

	def fit(self, rs: RecordSet) -> None:
		"""
		'Fits' the KNN to the provided dataset.

		In reality, the KNN does not really fit to the data, but simply stores
		the training observations for reference during prediction.

		:param rs: The record set to fit with.
		"""
		self.data = cp.deepcopy(rs)

	def point_prediction(self, point: np.ndarray) -> np.ndarray:
		"""
		Predicts the class label of a single point.

		The distance metric used is the L2-norm, but without the square root
		for a slight computational advantage.

		:param point: A row vector, with one feature per column.
		:return: A two-dimensional vector of 1-by-1. The class label.
		"""
		dists: np.ndarray = self.data.entries[:, :-1] - point
		dists = np.sum(dists ** 2, axis=1, keepdims=True)
		with_labels: np.ndarray = np.concatenate((dists, self.data.entries[:, [-1]]), axis=1)
		with_labels = with_labels[with_labels[:, 0].argsort()]
		first_k: np.ndarray = with_labels[:self.k, [1]]
		return st.mode(first_k)[0]  # choose most-occurring neighbor type

	def predict(self, rs: RecordSet) -> np.ndarray:
		"""
		Assigns a predicted class label to the given record sets.

		:param rs: The record set to assign predictions to.
		:return: A column vector of predictions corresponding to the record set's rows.
		"""
		predictions: np.ndarray = np.zeros((rs.entries.shape[0], 1))
		for r in range(rs.entries.shape[0]):
			point: np.ndarray = rs.entries[[r], :-1]  # do not include the ground truths
			predictions[[r], :] = self.point_prediction(point)
		return predictions
