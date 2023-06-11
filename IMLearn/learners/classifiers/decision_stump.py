from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm
    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split
    self.j_ : int
        The index of the feature by which to split the data
    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a decision stump to the given data. That is, finds the best feature and threshold by which to split
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        feature_thresholds = np.ndarray((2, X.shape[1]))
        error_matrix = np.ndarray((2, X.shape[1]))

        for feature_index in range(X.shape[1]):
            for sign in [1, -1]:
                optimal_threshold, corresponding_error = self._find_threshold(X[:, feature_index], y, sign)

                row_index = 0 if sign == -1 else 1
                feature_thresholds[row_index, feature_index] = optimal_threshold
                error_matrix[row_index, feature_index] = corresponding_error

        min_error_index = np.unravel_index(np.argmin(error_matrix), error_matrix.shape)
        self.sign_ = -1 if min_error_index[0] == 0 else 1
        self.j_ = min_error_index[1]
        self.threshold_ = feature_thresholds[min_error_index]

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict sign responses for given samples using fitted estimator
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        conditions = [X[:, self.j_] < self.threshold_, X[:, self.j_] >= self.threshold_]
        choices = [-self.sign_, self.sign_]

        return np.select(conditions, choices)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature
        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for
        labels: ndarray of shape (n_samples,)
            The labels to compare against
        sign: int
            Predicted label assigned to values equal to or above threshold
        Returns
        -------
        thr: float
            Threshold by which to perform split
        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold
        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        ordered_indices = np.argsort(values)
        values, labels = values[ordered_indices], labels[ordered_indices]

        sample_count = values.shape[0]
        accumulated_errors = np.zeros(sample_count + 1)
        sign_vector = np.full(sample_count, sign)

        for i in range(sample_count + 1):
            accumulated_errors[i] = np.sum(np.where(sign_vector != np.sign(labels), np.abs(labels), 0))
            if i < sample_count:
                sign_vector[i] = -sign

        min_error_index = np.argmin(accumulated_errors)

        if min_error_index == sample_count :
            optimal_threshold = values[-1] + 0.1
        else:
            optimal_threshold = values[min_error_index]

        return optimal_threshold, accumulated_errors[min_error_index] / len(values)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples
        y : ndarray of shape (n_samples, )
            True labels of test samples
        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))

