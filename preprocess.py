from __future__ import annotations
import math
import copy as cp
import numpy as np
import pandas as pd
import random as rd
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
		for col in range(len(self.entries.shape[1])):
			if self.types[col] == np.dtype('bool') and not also_categorical:
				continue
			self.entries[:, col] -= self.entries[:, col].mean()

	def scale(self, also_categorical: bool = False) -> None:
		for col in range(len(self.entries.shape[1])):
			if self.types[col] == np.dtype('bool') and not also_categorical:
				continue
			self.entries[:, col] /= self.entries[:, col].var()

	def normalise(self, also_categorical: bool = False) -> None:
		# TODO: Important. Perhaps we first need to use PCA, and only then standardise.
		self.center(also_categorical)
		self.scale(also_categorical)


def raw_data(location: str) -> RecordSet:
	try:
		df: pd.DataFrame = pd.read_csv(location)
		return RecordSet(df)
	except FileNotFoundError:
		raise Exception('Could not find the file \'' + location + '\'.')


if __name__ == '__main__':
	rec: RecordSet = raw_data('data.csv')
