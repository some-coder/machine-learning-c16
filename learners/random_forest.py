import numpy as np
from preprocess import RecordSet
import copy as cp
from learners.learner import Learner
from preprocess import RecordSet
from typing import Optional
from sklearn.ensemble import RandomForestClassifier

import graphviz


class RandomForest:
    """
    Establishes a base class from which learning algorithms can extend.
    """

    def __init__(self, depth: int, **params: any) -> None:
        """
        Constructs a base learning algorithm.

         :param params: Parameter values to supply.
        """
        super().__init__(**params)

        self.data: Optional[RecordSet] = None
        self.random_forest = None
        self.tree_depth = depth

    def fit(self, rs: RecordSet) -> None:
        """
        Fits the base learning algorithm to training data.

        :param rs: The record set to provide as training input.
        """


        self.data = cp.deepcopy(rs)
        x = self.data.entries[:, :-1]
        y = np.ravel(self.data.entries[:, -1:])
        forest = RandomForestClassifier(max_depth=self.tree_depth, criterion="entropy")
        forest.fit(x, y)
        self.random_forest = forest

    def predict(self, rs: RecordSet) -> np.ndarray:
        """
        Makes the base learner predict validation or testing data.

        Two notes need to be made. First, the last column of the provided record set
        contains the ground truth information. These entries, of course, may not be
        used during prediction. Second, the output Numpy array needs to be a column
        vector of two-dimensions: the first (rows) is equal to the length of the number
        of records, and the second (columns) is equal to 1.

        :param rs: The record set to provide as validation or testing data.
        :return: The predictions. A column vector with as many rows as the record set has.
        """
        predictions: np.ndarray = np.zeros((rs.entries.shape[0], 1))
        for r in range(rs.entries.shape[0]):
            predictions[[r], :] = self.random_forest.predict(rs.entries[[r], :-1])
            #print(predictions[[r], :])
        return predictions
