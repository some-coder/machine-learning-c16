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

	def __init__(self, df: pd.DataFrame) -> None:
		self.entries: np.ndarray = df.to_numpy()
		self.names: List[str] = [c.lower() for c in df.columns]
		self.types = self.column_types(df)

	def column_types(self, df: pd.DataFrame) -> List[np.dtype]:
		tps: List[np.dtype] = list(df.dtypes)
		for col in range(len(tps)):
			# correct Pandas by re-checking whether an integer column is a boolean column
			t: np.dtype = tps[col]
			if t == np.dtype('int64') and self.entries[:, col].min() == 0 and self.entries[:, col].max() == 1:
				tps[col] = np.dtype('bool')
		return tps

	def partition(self, percent: float, rng: Optional[rd.Random] = None) -> Tuple[RecordSet, RecordSet]:
		split: int = math.floor(percent * self.entries.shape[0])
		indices: List[int] = list(range(self.entries.shape[0]))
		if rng is not None:
			rng.shuffle(indices)
		ra: RecordSet = cp.deepcopy(self)
		rb: RecordSet = cp.deepcopy(self)
		ra.entries = ra.entries[indices[:split], :]
		rb.entries = rb.entries[indices[split:], :]
		return ra, rb

	def center(self, also_categorical: bool = False) -> None:
		for col in range(self.entries.shape[1]):
			if self.types[col] == np.dtype('bool') and not also_categorical:
				continue
			self.entries[:, col] -= self.entries[:, col].mean()

	def scale(self, also_categorical: bool = False) -> None:
		for col in range(self.entries.shape[1]):
			if self.types[col] == np.dtype('bool') and not also_categorical:
				continue
			self.entries[:, col] /= self.entries[:, col].var()

	def normalise(self, also_categorical: bool = False) -> None:
		self.center(also_categorical)
		self.scale(also_categorical)


def raw_data(location: str) -> RecordSet:
	try:
		df: pd.DataFrame = pd.read_csv(location)
		return RecordSet(df)
	except FileNotFoundError:
		raise Exception('Could not find the file \'' + location + '\'.')


def apply_pca(r: RecordSet, dim: Optional[int]) -> PCA:
	inp: np.ndarray = r.entries[:, :(r.entries.shape[1] - 1)]
	out: np.ndarray = r.entries[:, (r.entries.shape[1] - 1):]
	pca: PCA = PCA(n_components=dim)
	pca.fit(inp)
	if dim is None:
		dim = pca.explained_variance_.shape[0]
	r.entries = np.concatenate((pca.transform(inp), out), axis=1)
	r.names = ['component_' + str(i + 1) for i in range(dim)] + [r.names[len(r.names) - 1]]
	r.types = [np.dtype('float64') for i in range(dim)] + [r.types[len(r.types) - 1]]
	return pca


def visualise_pca_components(pca: PCA, block: bool = True) -> None:
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
	rec: RecordSet = raw_data('data.csv')
	gen: rd.Random = rd.Random(123)  # for reproducibility
	tv, te = rec.partition(0.7, gen)  # two datasets, namely train-validate and test
	tv.normalise()
	pca = apply_pca(tv, None)
	visualise_pca_components(pca, block=True)
