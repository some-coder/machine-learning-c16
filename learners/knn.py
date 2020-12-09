from learners.learner import Learner


class KNN(Learner):

	def __init__(self, a: int, **params: any):
		super().__init__(**params)
