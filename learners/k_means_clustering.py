import numpy as np
import copy as cp

from learners.learner import Learner
from preprocess import RecordSet
from typing import Optional


class KMeansClustering(Learner):
	"""
	The K-means clustering learning algorithm. Given data, this class
	constructs and stores a set of codebook vectors that can be used to
	classify test data-points.
	"""

	def __init__(self, k: int, **params: any):
		"""
		Initialises the K-means clustering learning algorithm.

		:param k: The number of clusters for K-means. K can be larger than 2.
		:param params: Ignored.
		"""
		super().__init__(**params)
		self.name = 'K-Means Clustering'
		self.k = k
		self.final_number_of_clusters = None
		self.data: Optional[RecordSet] = None
		self.codebook_vectors = None

	def fit(self, rs: RecordSet) -> None:
		"""
		Performs KMC on the provided (train) dataset.

		Performs an assignment step and an update step until
		the data-points no longer change clusters

		:param rs: The record set to fit with.
		"""
		self.data = cp.deepcopy(rs)

		np.random.shuffle(self.data.entries)  # the data set is now randomly shuffled

		list_of_data_points = np.array_split(self.data.entries, self.k)  # the data set is now split into K clusters
		codebooks = []
		for cluster in list_of_data_points:
			codebooks.append(np.mean(cluster[:, :], axis=0))  # mean per column (including outcome)
		codebook_array = np.array(codebooks)
		old_codebook_array = codebook_array

		i = 0
		while i < 100:  # max 100 iterations before stopping
			new_list = [[] for _ in range(self.k)]
			for datapoint in self.data.entries:
				dist = codebook_array[:, :-1] - datapoint[:-1]
				dist = np.sqrt(np.sum(dist ** 2, axis=1))
				new_list[np.where(dist == np.amax(dist))[0][0]].append(datapoint)
			codebooks = []
			for cluster in new_list:
				try:
					cluster = np.stack(cluster)
				except ValueError:
					pass
				else:
					codebooks.append(np.mean(cluster[:, :], axis=0))  # mean per column (including outcome)
			codebook_array = np.array(codebooks)

			old = old_codebook_array[:, 0:-1]
			new = codebook_array[:, 0:-1]

			if np.array_equal(old[old[:, 1].argsort()], new[new[:, 1].argsort()]):
				break    # if the codebook vectors don't change anymore, we're done
			old_codebook_array = codebook_array
			i += 1
		self.final_number_of_clusters = len(codebook_array)
		self.codebook_vectors = codebook_array

	def predict(self, rs: RecordSet) -> np.ndarray:
		"""
		Assigns a predicted class label to the given record sets.

		:param rs: The record set to assign predictions to.
		:return: A column vector of predictions corresponding to the record set's rows.
		"""
		print("\tfinal number of clusters: ", self.final_number_of_clusters)
		predictions: np.ndarray = np.zeros((rs.entries.shape[0], 1))
		outcome_codebook = self.codebook_vectors[:, -1]
		highest_value = np.amax(outcome_codebook)   # binary classes: highest value is 1.
		outcome_codebook[outcome_codebook == highest_value] = 1   # conditional
		outcome_codebook[outcome_codebook < highest_value] = 0
		for r in range(rs.entries.shape[0]):
			datapoint: np.ndarray = rs.entries[[r], :-1]  # do not include the ground truths
			dist = self.codebook_vectors[:, :-1] - datapoint
			dist = np.sqrt(np.sum(dist ** 2, axis=1))

			predictions[[r], :] = outcome_codebook[np.where(dist == np.amax(dist))[0][0]]
		return predictions
