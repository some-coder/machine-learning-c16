import numpy as np
from learners.learner import Learner
from preprocess import RecordSet
from sklearn.neural_network import MLPClassifier


class NeuralNetwork(Learner):
	"""
	The neural network learning algorithm. Given data, this class
	constructs and stores an intricate network of artificial neurons
	that can be used to classify test data-points.
	"""

	def __init__(self, h: tuple, a: str, r: float, speed: float, stop: int, m: int, **params: any) -> None:
		"""
		Initialises the neural network algorithm (multilayer perceptron, fully connected).

		:param list h: List with size of the hidden layers. List length and size must be at least 1.
		:param a: Activation function to be used. Options are: "identity", "relu", "tanh", "logistic".
		:param r: L2 regularization term. Default value in sklearn is 0.0001.
		:param speed: Learning rate. Default value in sklearn is 0.001.
		:param stop: Maximum number of iterations. Default value in sklearn is 200.
		:param m: Momentum for the (s)gd update. Must be between 0 and 1.0. Default value in sklearn is 0.9.
		:param params: Ignored.

		the MLPClassifier class from the sklearn package contains many parameters. For the parameters that we do
		not set ourselves, we rely on the default values. Information on these default values can be found here:
		https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
		"""
		super().__init__(**params)
		self.name = 'Artificial Neural Network (multi-layer perceptron)'

		self.classifier: MLPClassifier = \
			MLPClassifier(
				hidden_layer_sizes=h,
				activation=a,
				alpha=r,
				learning_rate_init=speed,
				max_iter=stop,
				momentum=m,
				nesterovs_momentum=False)

	def fit(self, rs: RecordSet) -> None:
		"""
		Fits the base learning algorithm to training data.

		:param rs: The record set to provide as training input.
		"""
		x: np.ndarray = rs.entries[:, :-1]
		y: np.ndarray = rs.entries[:, -1]

		self.classifier.fit(x, y)

	def predict(self, rs: RecordSet) -> np.ndarray:
		"""
		Makes the base learner predict validation or testing data.

		:param rs: The record set to provide as validation or testing data.
		:return: The predictions. A column vector with as many rows as the record set has.
		"""
		x: np.ndarray = rs.entries[:, :-1]
		y: np.ndarray = self.classifier.predict(x)
		return y.reshape((y.shape[0], 1))
