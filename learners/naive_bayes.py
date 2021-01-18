import numpy as np
import copy as cp
from learners.learner import Learner
from preprocess import RecordSet
from typing import Optional
from sklearn.naive_bayes import GaussianNB, BernoulliNB


class NaiveBayes(Learner):
	"""
	The naive Bayes learning algorithm. Given data, this class
	constructs and stores a relatively efficient Bayesian classifier
	(due to 'naive' assumptions) that can be used to classify test
	data-points.
	"""

	def __init__(self, prior, **params: any) -> None:
		"""
        Constructs a base learning algorithm.

         :param params: Parameter values to supply.
        """
		super().__init__(**params)

		self.data: Optional[RecordSet] = None
		self.bayes_model = None
		# prior = prior probability of class = 0

		self.prior = prior

	def fit(self, rs: RecordSet) -> None:
		"""
        Fits the base learning algorithm to training data.

        :param rs: The record set to provide as training input.
        """
		self.data = cp.deepcopy(rs)
		x = self.data.entries[:, :-1]
		y = np.ravel(self.data.entries[:, -1:])
		if self.prior != -1:
			bayes_classifier = GaussianNB(priors=[self.prior, 1 - self.prior])
			#bayes_classifier = BernoulliNB(class_prior=[self.prior, 1 - self.prior])
		else:
			bayes_classifier = GaussianNB()
			#bayes_classifier = BernoulliNB()
		bayes_classifier.fit(x, y)
		self.bayes_model = bayes_classifier

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
		"""
        print("class prior", self.bayes_model.class_prior_)
        print("classes", self.bayes_model.classes_)
        print("variance", self.bayes_model.sigma_)
        print("mean", self.bayes_model.theta_)
        """
		for r in range(rs.entries.shape[0]):
			predictions[[r], :] = self.bayes_model.predict(rs.entries[[r], :-1])
		# print(predictions[[r], :])
		return predictions


