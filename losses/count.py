import numpy as np
from losses.loss import Loss


class CountLoss(Loss):

	@staticmethod
	def compute(actual: np.ndarray, expected: np.ndarray) -> float:
		cl: np.ndarray = np.abs(actual - expected)
		cl[cl > 0] = 1
		return cl.sum()
