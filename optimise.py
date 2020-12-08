import copy as cp
import itertools as it
import numpy as np
from typing import Dict, List, Tuple, Type
from learners.learner import Learner
from losses.loss import Loss
from preprocess import RecordSet


Parameter = str
Values = Tuple[any]
GridAxis = Tuple[Parameter, Values]
Grid = Tuple[GridAxis, ...]
ConfigEntry = Tuple[Parameter, any]
Config = Tuple[ConfigEntry, ...]
InExRange = Tuple[int, int]


def dict_from_tuple_of_tuples(tot: Tuple[Tuple[str, any], ...]) -> Dict[str, any]:
	"""
	Obtains a string-any dictionary from a tuple-of-tuples (TOT).

	:param tot: The TOT.
	:return: The dictionary derived from the TOT.
	"""
	d: Dict[str, any] = {}
	for i in range(len(tot)):
		key: str = tot[i][0]
		val: any = tot[i][1]
		d[key] = val
	return d


class Optimiser:
	"""
	Optimisers, using K-fold cross-validation, attempt to find
	risk-minimising configurations of learning algorithms by
	search through a pre-defined parameter grid.
	"""

	def __init__(self, rs: RecordSet, lrn: Type[Learner], grd: Grid, k: int, losses: Tuple[Loss]) -> None:
		"""
		Constructs an optimiser.

		:param rs: The train-validation subset of the complete, pre-processed dataset.
		:param lrn: A learning algorithm class (not an instance of it).
		:param grd: A parameter grid to search over.
		:param k: The number of folds to use.
		:param losses: Loss functions to use. Must be minimally one.
		"""
		self.all_data = rs
		self.all_configs = self.configs_from_grid(grd)
		self.n = self.all_data.entries.shape[0]
		self.folds = self.folds_from_data()
		self.lrn = lrn
		self.k = k
		self.losses = losses
		self.evs: np.ndarray = np.zeros(len(self.losses), len(self.all_configs))

	@staticmethod
	def configs_from_grid(grd: Grid) -> List[Config]:
		"""
		'Unfolds' a parameter grid into a sequence of single parameter configurations.

		Parameter order is determined by the order of the 'parameter axes' in the
		grid. For instance, if some parameter A with its values was listed before
		another parameter B, then single configurations list their A-value before
		their B-value.

		Parameter names themselves are not included in the unfurled grid.

		:param grd: The parameter grid to unfurl.
		:return: The unfurled parameter grid.
		"""
		axes_values: Tuple[Values] = tuple([ga[1] for ga in grd])
		return list(it.product(*axes_values))

	def folds_from_data(self) -> Tuple[InExRange]:
		"""
		Yields a series of in- and exclusion index ranges, defining the folding scheme.

		The indices are with respect to the rows of the train-validation
		record set; they are the 'patterns'.

		The folding scheme is used by the optimiser to realise its K-fold
		cross-validation method.

		:return: A sequence of in- and exclusion index ranges.
		"""
		f: List[InExRange] = []
		indices: List[int] = list(np.floor(np.arange(0, self.n, self.n / self.k)).astype(int))
		indices += [self.n - 1]
		for i in range(len(indices) - 1):
			f.append((indices[i], indices[i + 1]))
		return tuple(f)

	def evaluate(self, train: RecordSet, validate: RecordSet, c: Config) -> np.ndarray:
		"""
		Evaluates the model with the specified configuration, on this specific train-validation split.

		:param train: The training data.
		:param validate: The testing data.
		:param c: The configuration of the model to evaluate for.
		:return: A column vector of loss metric values.
		"""
		cd: Dict[str, any] = dict_from_tuple_of_tuples(c)
		model = self.lrn(**cd)
		model.fit(train)
		prd: np.ndarray = model.predict(validate)
		out: np.ndarray = np.zeros((len(self.losses), 1))
		for i in range(len(self.losses)):
			loss: Loss = self.losses[i]
			out[i] = loss.compute(prd, validate[:, -1])
		return out

	def evaluate_all(self) -> None:
		"""
		Evaluates the model on all configurations, over K folds.

		Loss metrics are noted per configuration, and averaged over
		folds. The resultant #losses-by-#configurations matrix is
		stored in the class, ready to be requested by other objects.
		"""
		train: RecordSet = cp.deepcopy(self.all_data)
		validate: RecordSet = cp.deepcopy(self.all_data)
		for i in range(len(self.folds)):
			# compute the train and validation sets for this fold
			start: int = self.folds[i][0]
			end: int = self.folds[i][1]
			train_ids: List[int] = list(range(start)) + list(range(end, self.n))
			validate_ids: List[int] = list(range(start, end))
			train.entries = self.all_data.entries[train_ids, :]
			validate.entries = self.all_data.entries[validate_ids, :]
			# go over the configurations in the grid, and evaluate
			for c in range(len(self.all_configs)):
				config: Config = self.all_configs[c]
				self.evs[:, c] += self.evaluate(train, validate, config)
		# take, per loss metric, the average over folds
		self.evs /= self.k * np.ones((len(self.losses), 1))
