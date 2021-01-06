import numpy as np
import copy as cp
from learners.learner import Learner
from preprocess import RecordSet
from typing import Optional
from sklearn import tree

import graphviz


class DecisionTree(Learner):
	"""
	The decision tree learning algorithm. Given data, this class
	constructs and stores a decision tree that can be used to
	classify test data-points.
	"""

	def __init__(self, **params: any) -> None:
		"""
		Constructs a base learning algorithm.

		:param params: Parameter values to supply.
		"""
		super().__init__(**params)
		self.name = 'Decision Tree'

		self.data: Optional[RecordSet] = None
		self.decision_tree = None

	def fit(self, rs: RecordSet) -> None:
		"""
		Fits the base learning algorithm to training data.

		:param rs: The record set to provide as training input.
		"""
		self.data = cp.deepcopy(rs)
		x = self.data.entries[:, :-1]
		y = self.data.entries[:, -1:]
		tree_classifier = tree.DecisionTreeClassifier(criterion="entropy")
		tree_classifier.fit(x, y)
		self.decision_tree = tree_classifier

	def predict(self, rs: RecordSet) -> np.ndarray:
		"""
		Makes the base learner predict validation or testing data.

		Two notes need to be made. First, the last column of the provided record set
		contains the ground truth information. These entries, of course, may not be
		used during prediction. Second, the output Numpy array needs to be a column
		vector of two-dimensions: the first (rows) is equal to the length of the number
		of records, and the second (columns) is equal to 1.

		:param rs: The record set to provide as validation or testing data.
		:return: The predictions. A column vector with as many rows as the record set has.
		"""
		predictions: np.ndarray = np.zeros((rs.entries.shape[0], 1))
		dot_file = tree.export_graphviz(self.decision_tree, out_file=None)
		graph = graphviz.Source(dot_file)
		graph.render()
		for r in range(rs.entries.shape[0]):
			predictions[[r], :] = self.decision_tree.predict(rs.entries[[r], :-1])
			print(predictions[[r], :])
		return predictions
