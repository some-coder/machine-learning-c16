from performances.performance_metric import PerformanceMetric
import numpy as np


class TruePositive(PerformanceMetric):

	@staticmethod
	def compute(actual: np.ndarray, expected: np.ndarray) -> float:
		return (expected[actual == 1]).sum()
