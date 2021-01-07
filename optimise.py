import copy as cp
import itertools as it
from learners.learner import Learner
from losses.loss import Loss
import numpy as np
from preprocess import RecordSet
from typing import Dict, List, Optional, Tuple, Type


Parameter = str
Values = Tuple[any]
GridAxis = Tuple[Parameter, Values]
Grid = Tuple[GridAxis, ...]
ConfigEntry = Tuple[Parameter, any]
Config = Tuple[ConfigEntry, ...]
InExRange = Tuple[int, int]


def dict_from_tuple_of_tuples(tot: Tuple[Tuple[str, any], ...]) -> Dict[str, any]:
	"""
	Obtains a string-to-any dictionary from a tuple-of-tuples (TOT).

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
	risk-minimising configurations of learning algorithms by a
	search through a pre-defined parameter grid.
	"""

	def __init__(self, rs: RecordSet, lrn: Type[Learner], grd: Grid, k: int, losses: Tuple[Loss]) -> None:
		"""
		Constructs an optimiser.

		:param rs: The train-validation subset of the complete, pre-processed dataset.
		:param lrn: A learning algorithm class. (Not an instance. Must be a subclass of Learner.)
		:param grd: A parameter grid to search over.
		:param k: The number of folds to use.
		:param losses: Loss functions to use. At least one must be given.
		"""
		self.all_data = rs
		self.lrn = lrn
		self.k = k
		self.losses = losses
		self.n = self.all_data.entries.shape[0]
		self.all_config_values = self.configs_from_grid(grd)
		self.grid_parameters: List[Parameter] = [grd[i][0] for i in range(len(grd))]
		self.evs: np.ndarray = np.zeros((len(self.losses), len(self.all_config_values)))
		self.folds = self.folds_from_data()

	@staticmethod
	def configs_from_grid(grd: Grid) -> List[Tuple[any, ...]]:
		"""
		'Unfolds' a parameter grid into a sequence of single parameter configurations.

		Parameter order is determined by the order of the 'parameter axes' in the
		grid. For instance, if some parameter A with its values was listed before
		another parameter B, then single configurations list their A-value before
		their B-value.

		Parameter names themselves are not included in the unfurled grid.

		:param grd: The parameter grid to unfurl.
		:return: The unfurled parameter grid, without parameter keys.
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
		indices: List[int] = list(np.floor(np.linspace(0, self.n, self.k + 1)).astype(int))
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
			out[i] = loss.compute(prd, validate.entries[:, [-1]])
		return out

	def evaluate_all(self, show: bool = False) -> None:
		"""
		Evaluates the model on all configurations, over K folds.

		Loss metrics are noted per configuration, and averaged over
		folds. The resultant #losses-by-#configurations matrix is
		stored in the class, ready to be requested by other objects.

		:param show: Whether to show evaluation happening at standard output.
		"""
		train: RecordSet = cp.deepcopy(self.all_data)
		validate: RecordSet = cp.deepcopy(self.all_data)
		for fold in self.folds:
			# compute the train and validation sets for this fold
			if show:
				print('Computing fold ' + fold.__str__() + '.')
			start: int = fold[0]
			end: int = fold[1]
			train_ids: List[int] = list(range(start)) + list(range(end, self.n))
			validate_ids: List[int] = list(range(start, end))
			train.entries = self.all_data.entries[train_ids, :]
			validate.entries = self.all_data.entries[validate_ids, :]
			# go over the configurations in the grid, and evaluate
			for c in range(len(self.all_config_values)):
				config_values: Tuple[any, ...] = self.all_config_values[c]
				config: Config = tuple([(self.grid_parameters[i], config_values[i]) for i in range(len(config_values))])
				if show:
					print('\tConfiguration ' + str(config) + '.')
				self.evs[:, [c]] += self.evaluate(train, validate, config)
		# take, per loss metric, the average over folds
		self.evs /= self.k * np.ones((len(self.losses), 1))

	def best_configurations(self, loss_index: int, n: Optional[int] = None) -> Tuple[Config]:
		"""
		Yields a top-N of configurations, based on a loss given by its index.

		Note that N must be at least one, and may be maximally equal to the number of
		configurations.

		:param loss_index: The index of the loss method to find best configurations for.
		:param n: Optional. The top-N of configurations to get; the rest is ignored.
		:return: A tuple of the top-N best configurations, sorted in descending order.
		"""
		with_ids: np.ndarray = np.concatenate((np.array([range(self.evs.shape[1])]), self.evs), axis=0)
		ordered: np.ndarray = with_ids[:, self.evs[loss_index, :].argsort()]
		best_n: List[int] = list(ordered[0, :(n if n is not None else ordered.shape[1])].astype('int'))
		best_configs: List[Config] = []
		for i in best_n:
			good_config_values: Tuple[any, ...] = self.all_config_values[i]
			good_config: Config = \
				tuple([(self.grid_parameters[i], good_config_values[i]) for i in range(len(good_config_values))])
			best_configs.append(good_config)
		return tuple(best_configs)
