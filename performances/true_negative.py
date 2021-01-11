from performances.performance_metric import PerformanceMetric
import numpy as np


class TrueNegative(PerformanceMetric):

	@staticmethod
	def compute(actual: np.ndarray, expected: np.ndarray) -> float:
		return ((np.logical_not(expected))[actual != 1]).sum()
