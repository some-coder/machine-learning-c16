from __future__ import annotations
import math
import copy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rd
from sklearn.decomposition import PCA
from typing import List, Optional, Tuple


class RecordSet:
	"""
	RecordSets serve as convenience object around the data of this project.

	Using RecordSets is preferable over Numpy arrays or Pandas dataframes:
	the true data types of columns are stored, as well as their names;
	RecordSets can easily be partitioned for train-test splitting, and
	RecordSets have in-built normalisation.
	"""

	def __init__(self, df: pd.DataFrame) -> None:
		"""
		Constructs a RecordSet.

		:param df: A Pandas dataframe with the raw records.
		"""
		self.entries: np.ndarray = df.to_numpy()
		self.names: List[str] = [c.lower() for c in df.columns]
		self.types = self.column_types(df)

	def column_types(self, df: pd.DataFrame) -> List[np.dtype]:
		"""
		Yields the Numpy data types associated with the columns of the Pandas dataframe.

		:param df: A Pandas dataframe with the raw records.
		:return: The Numpy data types of the dataframe's columns.
		"""
		tps: List[np.dtype] = list(df.dtypes)
		for col in range(len(tps)):
			# correct Pandas by re-checking whether an integer column is a boolean column
			t: np.dtype = tps[col]
			if t == np.dtype('int64') and self.entries[:, col].min() == 0 and self.entries[:, col].max() == 1:
				tps[col] = np.dtype('bool')
		return tps

	def partition(self, percent: float, rng: Optional[rd.Random] = None) -> Tuple[RecordSet, RecordSet]:
		"""
		Separates this set of records into two subsets.

		If no random number generator is supplied, the first N rows are used
		for the first subset, where N is greatest integer smaller than or
		equal to the percentage of the total number of rows.

		:param percent: The percentage of rows to assign to the first subset.
		:param rng: If supplied: the RNG for selecting rows for the first subset.
		:return: A partitioning of this record set into two subsets.
		"""
		split: int = math.floor(percent * self.entries.shape[0])
		indices: List[int] = list(range(self.entries.shape[0]))
		if rng is not None:
			rng.shuffle(indices)
		ra: RecordSet = cp.deepcopy(self)
		rb: RecordSet = cp.deepcopy(self)
		ra.entries = ra.entries[indices[:split], :]
		rb.entries = rb.entries[indices[split:], :]
		return ra, rb

	def center(
			self,
			pre_calc_mu: Optional[np.ndarray] = None,
			also_categorical: bool = False,
			also_output: bool = True) -> None:
		"""
		Centers columns around their (sample) mean value.

		:param pre_calc_mu: Optional. A row vector of per-column means. If supplied, it will be used in computation.
		:param also_categorical: Whether to also center categorical columns. Defaults to no.
		:param also_output: Whether to also center the output column. Defaults to yes.
		"""
		idx: int = 0
		for col in range(self.entries.shape[1] - (0 if also_output else 1)):
			if self.types[col] == np.dtype('bool') and not also_categorical:
				continue
			if pre_calc_mu is not None:
				self.entries[:, col] -= pre_calc_mu[0, idx]
			else:
				self.entries[:, col] -= self.entries[:, col].mean()
			idx += 1

	def scale(self,
			pre_calc_std: Optional[np.ndarray] = None,
			also_categorical: bool = False,
			also_output: bool = True) -> None:
		"""
		Scales columns to have unit variance.

		:param pre_calc_std: Optional. A row vector of per-column SDs. If supplied, it will be used in computation.
		:param also_categorical: Whether to also scale categorical columns.
		:param also_output: Whether to also scale the output column. Defaults to yes.
		"""
		idx: int = 0
		for col in range(self.entries.shape[1] - (0 if also_output else 1)):
			if self.types[col] == np.dtype('bool') and not also_categorical:
				continue
			if pre_calc_std is not None:
				self.entries[:, col] /= pre_calc_std[0, idx]
			else:
				self.entries[:, col] /= self.entries[:, col].std()
			idx += 1

	def normalise(
			self,
			pre_calc_mu: Optional[np.ndarray] = None,
			pre_calc_std: Optional[np.ndarray] = None,
			also_categorical: bool = False,
			also_output: bool = True) -> None:
		"""
		Centers and scales columns.

		:param pre_calc_mu: Optional. A row vector of per-column means. If supplied, it will be used in computation.
		:param pre_calc_std: Optional. A row vector of per-column SDs. If supplied, it will be used in computation.
		:param also_categorical: Whether to also normalise categorical columns.
		:param also_output: Whether to also normalise the output column. Defaults to yes.
		"""
		self.center(pre_calc_mu, also_categorical, also_output)
		self.scale(pre_calc_std, also_categorical, also_output)


def raw_data(location: str) -> RecordSet:
	"""
	Yields raw record set data, given the source location.

	:param location: Absolute or relative path to the CSV containing the raw data.
	:return: The raw record set.
	"""
	try:
		df: pd.DataFrame = pd.read_csv(location)
		return RecordSet(df)
	except FileNotFoundError:
		raise Exception('Could not find the file \'' + location + '\'.')


def apply_pca(r: RecordSet, dim: Optional[int]) -> PCA:
	"""
	Performs linear PCA on a record set.

	This method performs PCA on the record set provided to it. After this is done,
	it returns the Scikit-Learn PCA object that conducted the analysis. This is
	useful when needing to project new patterns to the same feature subspace,
	as is the case when test data comes along.

	Slightly inconveniently, you need to specify the number of components to keep
	before conducting the PCA. Another method in this module allows you to plot
	the Variance Accounted For (VAF) per component.

	:param r: The record set to subject to a PC analysis.
	:param dim: The number of dimensions feature vectors of the PCA should have.
	:return: A Scikit-Learn PCA object for possible re-use.
	"""
	inp: np.ndarray = r.entries[:, :(r.entries.shape[1] - 1)]
	out: np.ndarray = r.entries[:, (r.entries.shape[1] - 1):]
	pca: PCA = PCA(n_components=dim)
	pca.fit(inp)
	if dim is None:
		dim = pca.explained_variance_.shape[0]
	r.entries = np.concatenate((pca.transform(inp), out), axis=1)
	r.names = ['component_' + str(i + 1) for i in range(dim)] + [r.names[len(r.names) - 1]]
	r.types = [np.dtype('float64') for _ in range(dim)] + [r.types[len(r.types) - 1]]
	return pca


def apply_pca_indirectly(r: RecordSet, pca: PCA) -> None:
	"""
	Performs linear PCA on a record set.

	In contrast to the direct PCA application method, defined above, this method
	allows you to apply PCA indirectly, using results from a previous PC analysis
	of another dataset. This is primarily useful when you have test data, that
	needs to be transformed similarly as previous training/validation data.

	This method does not return a PCA on its own; it simply 'piggy-backs' from
	a 'true' PCA, often performed on training data beforehand.

	Important. This method silently assumes that the RecordSet has been normalised
	(both in terms of centering, as well as scaling) in exactly the same way as
	was done for the dataset that yielded the PCA (`pca`) that is now used.

	:param r: The record set to subject to a PC analysis.
	:param pca: A Scikit-Learn PCA object of which to use the principal components matrix, U.
	"""
	new_patterns: np.ndarray = (pca.components_ @ r.entries[:, :-1].T).T  # excludes the output column
	r.entries = np.concatenate((new_patterns, r.entries[:, [-1]]), axis=1)  # include it


def visualise_pca_components(pca: PCA, block: bool = True) -> None:
	"""
	Visualises the Variance Accounted For (VAF) per component of a PC analysis.

	:param pca: A fitted Scikit-Learn PCA object.
	:param block: Whether to pause program execution when the plot is shown.
	"""
	# gather the bar values
	n: int = pca.explained_variance_.shape[0]
	x: List[int] = list(range(n))
	y: np.ndarray = pca.explained_variance_
	# make the actual plot
	plt.bar(x, y, color='black')
	plt.xticks(list(range(n)))
	plt.xlabel('PCA component')
	plt.ylabel('Variance accounted for (VAF)')
	plt.title('VAF per PCA Component Used')
	plt.tight_layout()
	plt.show(block=block)


if __name__ == '__main__':
	# perform PCA on a train-validate subset of the complete data, comprising 70% of it
	rec: RecordSet = raw_data('data.csv')
	gen: rd.Random = rd.Random(123)  # for reproducibility
	tv, te = rec.partition(0.7, gen)  # two datasets, namely train-validate and test
	tv_mu: np.ndarray = tv.entries.mean(axis=0).reshape((1, tv.entries.shape[1]))
	tv_std: np.ndarray = tv.entries.std(axis=0).reshape((1, tv.entries.shape[1]))
	tv.normalise(tv_mu, tv_std, True)
	tv_pca = apply_pca(tv, None)  # Choose None to include all PCs. (Is not the same as not doing the PCA.)
	te.normalise(tv_mu, tv_std, True)
	apply_pca_indirectly(te, tv_pca)
	visualise_pca_components(tv_pca, block=True)
