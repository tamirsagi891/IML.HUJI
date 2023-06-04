from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator
    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data
    X: ndarray of shape (n_samples, n_features)
       Input data to fit
    y: ndarray of shape (n_samples, )
       Responses of input data to fit to
    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.
    cv: int
        Specify the number of folds.
    Returns
    -------
    train_score: float
        Average train score over folds
    validation_score: float
        Average validation score over folds
    """
    X_folds = np.array_split(X, cv)
    y_folds = np.array_split(y, cv)

    scores_train, scores_val = np.zeros(cv), np.zeros(cv)

    for k in range(cv):
        train_X = np.vstack([fold for j, fold in enumerate(X_folds) if k != j])
        train_y = np.concatenate([fold for j, fold in enumerate(y_folds) if k != j])

        estimator.fit(train_X, train_y)

        scores_train[k] = scoring(train_y, estimator.predict(train_X))
        scores_val[k] = scoring(y_folds[k], estimator.predict(X_folds[k]))

    return scores_train.mean(), scores_val.mean()
