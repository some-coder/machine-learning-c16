from learners.learner import Learner
from learners.k_means_clustering import KMeansClustering
from learners.knn import KNN
from learners.linear_reg import LinearRegression
from learners.logistic_reg import LogisticRegression
from learners.naive_bayes import NaiveBayes
from learners.neural_network import NeuralNetwork
from learners.probit_reg import ProbitRegression
from learners.poisson_reg import PoissonRegression
from learners.random_forest import RandomForest
from learners.svm import SVM
from losses.loss import Loss
from losses.count import CountLoss
from optimise import Config, dict_from_tuple_of_tuples, Grid, Optimiser
from os import mkdir
from os.path import isfile
from performances.performance_metric import PerformanceMetric
from performances.true_positive import TruePositive
from performances.true_negative import TrueNegative
from performances.false_positive import FalsePositive
from performances.false_negative import FalseNegative
from preprocess import apply_pca, apply_pca_indirectly, raw_data, RecordSet
from random import Random, randint
from typing import cast, Dict, List, Optional, Tuple, Type
import numpy as np
import pandas as pd
import sys
import os
import warnings

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
		show: bool = False) -> Dict[Type[Learner], Tuple[Config, ...]]:
	"""
	Yields, for each model, the best configuration of its parameter grid.

	:param rs: The training-validation data. Specifically NOT the testing data!
	:param ms: A tuple of model-parameter grid pairs.
	:param k_fold_k: The number of folds to use. Minimally 1. Maximally the number of data-points (LOO-KCV).
	:param lrn: The loss metrics to employ. Use at least 1. Only the first loss metric is considered.
	:param show: Whether to show details at standard output. Defaults to no.
	:return: dictionary mapping models to configurations, decreasing in successfulness.
	"""
	total: Dict[Type[Learner], Tuple[Config, ...]] = {}
	for mod_grd in ms:
		model, grid = mod_grd
		if show:
			print('\t\tFinding best configurations for ' + model.__name__ + '.')
		opt: Optimiser = Optimiser(rs, model, grid, k_fold_k, lrn)
		opt.evaluate_all()
		bests: Tuple[Config, ...] = opt.best_configurations(loss_index=0)  # only consider first loss metric
		if show:
			print('\t\tBest configurations:')
			for rank, best in enumerate(bests):
				print('\t\t(' + str(rank) + ') ' + str(best))
		total[model] = bests
	return total


def test_outcomes(
		models: Tuple[Type[Learner], ...],
		configs: Tuple[Config, ...],
		tv: RecordSet,
		te: RecordSet,
		ls: Tuple[Loss, ...],
		pf: Tuple[PerformanceMetric, ...],
		show: bool = False) -> pd.DataFrame:
	"""
	Trains models on all of the train-validation data, and tests them on the testing data.

	:param models: The model classes to consider.
	:param configs: Per model, a parameter configuration.
	:param ls: A set of losses to apply during testing.
	:param pf: A set of performances to compute during testing.
	:param show: Whether to show details at standard output. Defaults to no.
	:return: Model performances on the testing data in a Pandas dataframe.
	"""
	test_losses: np.ndarray = np.zeros((len(ls), len(models)))
	test_performances: np.ndarray = np.zeros((len(pf), len(models)))
	model_name_list: list = list()
	model_config_list: list = list()
	for i, model in enumerate(models):
		if show:
			print('Applying test data to model ' + model.__name__ + '.')
		model_name_list.append(str(model.__name__))
		cd: Dict[str, any] = dict_from_tuple_of_tuples(configs[i])
		model_config_list.append(str(cd))
		mdl: Learner = model(**cd)
		mdl.fit(tv)
		prd: np.ndarray = mdl.predict(te)
		for j, loss in enumerate(ls):
			test_losses[j, i] = loss.compute(prd, te.entries[:, [-1]])
		for j, performance in enumerate(pf):
			test_performances[j, i] = performance.compute(prd, te.entries[:, [-1]])

	# add model names and put in pandas dataframe
	pd_loss: pd.DataFrame = pd.DataFrame(data=model_name_list)
	pd_loss.columns = ["model_name"]
	pd_loss["config"] = model_config_list
	pd_loss["loss"] = test_losses.transpose()
	pd_performances = pd.DataFrame(data=test_performances)
	pd_performances = pd_performances.transpose()
	pd_performances.columns = ["TP", "TN", "FP", "FN"]
	pd_res = pd.concat([pd_loss, pd_performances], axis=1)
	pd_res["accuracy"] = \
		(pd_res.loc[:, "TP"] + pd_res.loc[:, "TN"]) / \
		(pd_res.loc[:, "TP"] + pd_res.loc[:, "TN"] + pd_res.loc[:, "FP"] + pd_res.loc[:, "FN"])

	pd_res["F1"] = \
		(2 * pd_res.loc[:, "TP"]) / \
		(2 * pd_res.loc[:, "TP"] + pd_res.loc[:, "FP"] + pd_res.loc[:, "FN"])

	pd_res["MMC"] = \
		(pd_res.loc[:, "TP"] + pd_res.loc[:, "FP"]) * \
		(pd_res.loc[:, "TP"] + pd_res.loc[:, "FN"]) * \
		(pd_res.loc[:, "TN"] + pd_res.loc[:, "FP"]) * \
		(pd_res.loc[:, "TN"] + pd_res.loc[:, "FN"])
	pd_res.loc[:, "MMC"] = \
		((pd_res.loc[:, "TP"] * pd_res.loc[:, "TN"]) - (pd_res.loc[:, "FP"] * pd_res.loc[:, "FN"])) / \
		pd_res.loc[:, "MMC"] ** (1 / 2)
	return pd_res


if __name__ == '__main__':
	# suppress all warnings
	if not sys.warnoptions:
		warnings.simplefilter("ignore")
		os.environ["PYTHONWARNINGS"] = "ignore"

	# set paramaters
	see_test_performance: bool = True  # Only activate once we're sure of our models!
	df: Optional[pd.DataFrame] = None
	out_dir, out_file_name = 'results', 'all_best_models.csv'
	rng_grand: Random = Random(123456)
	replications: int = 100

	# loop S replications
	for replication in range(1, (replications + 1), 1):
		seed = rng_grand.randint(1, replication * 100)
		print(f'ITERATION {replication}')

		for num_pca_comps in range(7, 13, 1):
			print('PCA USES %d COMPONENTS' % (num_pca_comps,))
			#rng: Random = Random(22272566)  # For reproducibility. Different sample, better performance.
			rng: Random = Random(seed)  # For reproducibility. Different sample, better performance.
			dat_loc: str = 'data.csv'
			split_percent: float = 0.7
			pca_comps: int = num_pca_comps
			losses: Tuple[Loss, ...] = (CountLoss(),)
			performances: Tuple[PerformanceMetric, ...] = \
				(TruePositive(), TrueNegative(), FalsePositive(), FalseNegative())
			k: int = 10

			# learning algorithms and their parameter grids
			model_grids: Tuple[LearnerParameters, ...] = \
				cast(Tuple[LearnerParameters, ...], (
					(SVM, (
						('alpha', ([1.5])),
						('kernel', (['linear'])),
						('shrinking', ([False]))
					)),
					(LinearRegression, (('alpha', ([0.99])),)),
					(LogisticRegression, (('alpha', ([1.5])),)),
					(ProbitRegression, (('alpha', ([3.2])),)),
					(PoissonRegression, (('alpha', ([x / 10.0 for x in range(8, 17, 1)])),)),
					(KMeansClustering, (('k', ([2])),)),
					(KNN, (('k', ([4])),))#,
					(RandomForest, (('depth', ([4])),))#,
					(NaiveBayes, (('prior', ([0.48])),)),
					(NeuralNetwork, (
					 	('h', ([[8]])),
					 	('a', (['logistic'])),
					 	('alpha', ([0.0])),
					 	('speed', (0.01,)),
					 	('stop', (600,)),
					 	('m', ([0.9])),))
				))

			# find the best configurations per model
			print('\tFINDING OPTIMAL PARAMETERS')
			train_validate, test = preprocessed_data(dat_loc, split_percent, pca_comps, rng)
			b = best_configurations(train_validate, model_grids, k, losses, show=True)

			# report train-validation configuration results
			print('\n\tOPTIMAL PARAMETERS')
			for index, model_grid in enumerate(model_grids):
				for index_j, configuration in enumerate(b[model_grid[0]]):
					print('\t\t' + model_grid[0].__name__ + ': ' + str(configuration) + '.')

					# compute and output testing results
					if see_test_performance:
						pd_result = test_outcomes(
							models=tuple([model_grid[0]]),
							configs=tuple([configuration]),
							tv=train_validate,
							te=test,
							ls=losses,
							pf=performances,
							show=False
						)
						pd_result.insert(2, 'iteration', replication)
						pd_result.insert(3, 'm', num_pca_comps)
						if df is None:
							df = pd_result
						else:
							df = pd.concat([df, pd_result], ignore_index=True)

			# report that we're done
			print('PCA FOR %d COMPONENTS DONE\n' % (num_pca_comps,))

	if see_test_performance:
		save_mean = True
		print('\nRESULTS')

		if save_mean:
			df = df.drop(columns=['iteration'])
			df = df.groupby(["model_name", "config", "m"]).mean()

		df = df.sort_values('accuracy', ascending=False)
		print(df.to_string())

		print('\nWRITING... ', end='')
		#if not isfile(out_dir + '/' + out_file_name):  # still needs to be created
		#	try:
		#		mkdir(out_dir)
		#	except FileExistsError:
		#		raise Exception('\nFAILED: DIRECTORY ALREADY EXISTS!')
		df.to_csv(out_dir + '/' + out_file_name, index=True)
		print('DONE')
