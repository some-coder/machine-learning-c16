import numpy as np
from preprocess import RecordSet


class Learner:
	"""
	Establishes a base class from which learning algorithms can extend.
	"""

	def __init__(self, **params: any) -> None:
		"""
		Constructs a base learning algorithm.

		:param params: Parameter values to supply.
		"""
		pass

	def fit(self, rs: RecordSet) -> None:
		"""
		Fits the base learning algorithm to training data.

		:param rs: The record set to provide as training input.
		"""
		pass

	def predict(self, rs: RecordSet) -> np.ndarray:
		"""
		Makes the base learner predict validation or testing data.

		:param rs: The record set to provide as validation or testing data.
		:return: The predictions. A column vector with as many rows as the record set has.
		"""
		pass
