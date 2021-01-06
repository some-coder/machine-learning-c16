from learners.learner import Learner
from learners.k_means_clustering import KMeansClustering
from losses.loss import Loss
from losses.count import CountLoss
from optimise import Config, Grid, Optimiser
from preprocess import apply_pca, apply_pca_indirectly, raw_data, RecordSet
from random import Random
from typing import cast, List, Tuple, Type


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
	tv, te = rs.partition(0.7)
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
		k: int,
		lrn: Tuple[Loss, ...],
		show: bool = False) -> Tuple[Config, ...]:
	"""
	Yields, for each model, the best configuration of its parameter grid.

	:param rs: The training-validation data. Specifically NOT the testing data!
	:param ms: A tuple of model-parameter grid pairs.
	:param k: The number of folds to use. Minimally 1. Maximally the number of data-points (LOO-KCV).
	:param lrn: The loss metrics to employ. Use at least 1. Only the first loss metric is considered.
	:param show: Whether to show details at standard output. Defaults to no.
	:return: A tuple of, per model, the single best configuration.
	"""
	total: List[Config] = []
	for model_grid in ms:
		model, grid = model_grid
		print('Finding best configurations for ' + model.__name__ + '.')
		opt: Optimiser = Optimiser(rs, model, grid, k, lrn)
		opt.evaluate_all()
		bests: Tuple[Config, ...] = opt.best_configurations(loss_index=0)  # only consider first loss metric
		if show:
			print('Best configurations:')
			for rank, best in enumerate(bests):
				print('\t(' + str(rank) + ') ' + str(best))
		total.append(bests[0])
	return tuple(total)


if __name__ == '__main__':
	# some hyper-parameters of the system
	rng: Random = Random(123)  # for reproducibility
	dat_loc: str = 'data.csv'
	split_percent: float = 0.7
	pca_comps: int = 2
	losses: Tuple[Loss, ...] = (CountLoss(),)
	k: int = 10

	# learning algorithms and their parameter grids
	model_grids: Tuple[LearnerParameters, ...] = \
		cast(Tuple[LearnerParameters, ...],
			(
				(KMeansClustering, (('k', (1, 2, 3, 4)), )),
			))

	# the actual procedure
	print('FINDING OPTIMAL PARAMETERS')
	train_validate, test = preprocessed_data(dat_loc, split_percent, pca_comps, rng)
	b = best_configurations(train_validate, model_grids, k, losses, show=True)

	# report results
	print('\nRESULT')
	for index, model_grid in enumerate(model_grids):
		print('\t' + model_grid[0].__name__ + ': ' + str(b[index]) + '.')
