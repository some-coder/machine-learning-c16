from __future__ import annotations
import math
import copy as cp
import numpy as np
import pandas as pd
import random as rd
from typing import List, Optional, Tuple


class Records:

	def __init__(self, df: pd.DataFrame) -> None:
		self.entries: np.ndarray = df.to_numpy()
		self.names: List[str] = [c.lower() for c in df.columns]

	def partition(self, percent: float, rng: Optional[rd.Random] = None) -> Tuple[Records, Records]:
		split: int = math.floor(percent * self.entries.shape[0])
		indices: List[int] = list(range(self.entries.shape[0]))
		if rng is not None:
			rng.shuffle(indices)
		ra: Records = cp.deepcopy(self)
		rb: Records = cp.deepcopy(self)
		ra.entries = ra.entries[indices[:split], :]
		rb.entries = rb.entries[indices[split:], :]
		return ra, rb


def raw_data(location: str) -> Records:
	try:
		df: pd.DataFrame = pd.read_csv(location)
		return Records(df)
	except FileNotFoundError:
		raise Exception('Could not find the file \'' + location + '\'.')


if __name__ == '__main__':
	rec: Records = raw_data('data.csv')
	gen: rd.Random = rd.Random()
