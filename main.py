from learners.learner import Learner
from learners.k_means_clustering import KMeansClustering
from learners.knn import KNN
from learners.linear_reg import LinearRegression
from learners.logistic_reg import LogisticRegression
from learners.naive_bayes import NaiveBayes
from learners.neural_network import NeuralNetwork
from learners.probit_reg import ProbitRegression
from learners.random_forest import RandomForest
from learners.svm import SVM
from losses.loss import Loss
from losses.count import CountLoss
from optimise import Config, dict_from_tuple_of_tuples, Grid, Optimiser
from preprocess import apply_pca, apply_pca_indirectly, raw_data, RecordSet
from random import Random
from typing import cast, Dict, List, Tuple, Type
import numpy as np


LearnerParameters = Tuple[Type[Learner], Grid]


def preprocessed_data(loc: str, percent: float, m: int, gen: Random) -> Tuple[RecordSet, RecordSet]:
	"""
	Yields the train-and-validation and testing datasets.

	The two datasets have already been normalised (centered, scaled).
	Additionally, PCA has already been applied on them, as well.

	:param loc: An absolute or relative file location of the data. A CSV.
	:param percent: The splitting percentage. The higher, the more data goes to train-validation.
	:param m: The number of principal components to keep.
	:param gen: An RNG. Used for splitting up the data in two.
	:return: The two datasets, delivered as a two-tuple.
	"""
	rs: RecordSet = raw_data(loc)
	tv, te = rs.partition(percent, gen)
	mu = tv.entries.mean(axis=0).reshape((1, tv.entries.shape[1]))
	sd = tv.entries.std(axis=0).reshape((1, tv.entries.shape[1]))
	tv.normalise(mu, sd, also_categorical=True, also_output=False)
	pca = apply_pca(tv, m)
	te.normalise(mu, sd, also_categorical=True, also_output=False)
	apply_pca_indirectly(te, pca)
	return tv, te


def best_configurations(
		rs: RecordSet,
		ms: Tuple[LearnerParameters, ...],
		k_fold_k: int,
		lrn: Tuple[Loss, ...],
		show: bool = False) -> Tuple[Config, ...]:
	"""
	Yields, for each model, the best configuration of its parameter grid.

	:param rs: The training-validation data. Specifically NOT the testing data!
	:param ms: A tuple of model-parameter grid pairs.
	:param k_fold_k: The number of folds to use. Minimally 1. Maximally the number of data-points (LOO-KCV).
	:param lrn: The loss metrics to employ. Use at least 1. Only the first loss metric is considered.
	:param show: Whether to show details at standard output. Defaults to no.
	:return: A tuple of, per model, the single best configuration.
	"""
	total: List[Config] = []
	for mod_grd in ms:
		model, grid = mod_grd
		if show:
			print('Finding best configurations for ' + model.__name__ + '.')
		opt: Optimiser = Optimiser(rs, model, grid, k_fold_k, lrn)
		opt.evaluate_all()
		bests: Tuple[Config, ...] = opt.best_configurations(loss_index=0)  # only consider first loss metric
		if show:
			print('Best configurations:')
			for rank, best in enumerate(bests):
				print('\t(' + str(rank) + ') ' + str(best))
		total.append(bests[0])
	return tuple(total)


def test_outcomes(
		models: Tuple[Type[Learner], ...],
		configs: Tuple[Config, ...],
		tv: RecordSet,
		te: RecordSet,
		ls: Tuple[Loss, ...],
		show: bool = False) -> np.ndarray:
	"""
	Trains models on all of the train-validation data, and tests them on the testing data.

	The output is a matrix of #losses-by-#models.

	:param models: The model classes to consider.
	:param configs: Per model, a parameter configuration.
	:param ls: A set of losses to apply during testing.
	:param show: Whether to show details at standard output. Defaults to no.
	:return: Model performances on the testing data.
	"""
	performances: np.ndarray = np.zeros((len(ls), len(models)))
	for i, model in enumerate(models):
		if show:
			print('Applying test data to model ' + model.__name__ + '.')
		cd: Dict[str, any] = dict_from_tuple_of_tuples(configs[i])
		m: Learner = model(**cd)
		m.fit(tv)
		prd: np.ndarray = m.predict(te)
		for j, loss in enumerate(ls):
			performances[j, i] = loss.compute(prd, te.entries[:, [-1]])
	return performances


if __name__ == '__main__':
	# some hyper-parameters of the system
	rng: Random = Random(123)  # for reproducibility
	dat_loc: str = 'data.csv'
	split_percent: float = 0.7
	pca_comps: int = 2
	losses: Tuple[Loss, ...] = (CountLoss(),)
	k: int = 10
	see_test_performance: bool = False  # Only activate once we're sure of our models!

	# learning algorithms and their parameter grids
	model_grids: Tuple[LearnerParameters, ...] = \
		cast(Tuple[LearnerParameters, ...], (
				(KMeansClustering, (('k', (1, 2, 3)),)),
				(KNN, (('k', (1, 2, 3)),)),
				(LinearRegression, (('alpha', (0, 0.009, 0.01)),)),
				(LogisticRegression, (('alpha', (0.1, 3.5, 5)),)),
				(NaiveBayes, (('prior', (0.01, 0.5, 0.99, -1)),)),
				# (NeuralNetwork, ())  # Winward: Set neural network grid search parameters.
				(ProbitRegression, (('alpha', (0.1, 3.5, 5)),)),
				(RandomForest, (('depth', (2, 5, 3)),)),
				(SVM, (('C', (0.1, 0.5, 0.9, 1.0)),))
			))

	# find the best configurations per model
	print('FINDING OPTIMAL PARAMETERS')
	train_validate, test = preprocessed_data(dat_loc, split_percent, pca_comps, rng)
	b = best_configurations(train_validate, model_grids, k, losses, show=True)

	# report train-validation configuration results
	print('\nOPTIMAL PARAMETERS')
	for index, model_grid in enumerate(model_grids):
		print('\t' + model_grid[0].__name__ + ': ' + str(b[index]) + '.')

	# compute and output testing results
	if see_test_performance:
		print('\nAPPLYING TO TESTING DATA')
		out: np.ndarray = test_outcomes(
				models=tuple([mg[0] for mg in model_grids]),
				configs=tuple(b[best_i] for best_i in range(len(model_grids))),
				tv=train_validate,
				te=test,
				ls=losses,
				show=False
			)
		print('\nRESULTS')
		print(out)
