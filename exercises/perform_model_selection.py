from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree
    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate
    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    raise NotImplementedError()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    raise NotImplementedError()


def compute_errors(reg, params, train_X, train_y):
    train_errors, validation_errors = [], []

    if params is None:  # Case for LinearRegression
        model = reg()
        train_error, validation_error = cross_validate(
            model, train_X, train_y, mean_square_error, 5)
        train_errors.append(train_error)
        validation_errors.append(validation_error)
    else:
        for p in params:
            if p == 0:
                continue
            model = reg(p, True) if reg == RidgeRegression else reg(alpha=p, tol=0.001)
            train_error, validation_error = cross_validate(
                model, train_X, train_y, mean_square_error, 5)
            train_errors.append(train_error)
            validation_errors.append(validation_error)

    return train_errors, validation_errors


def plot_errors(fig, model_names, reg_params, train_errors, validation_errors):
    for idx, name in enumerate(model_names):
        fig.add_traces([go.Scatter(x=reg_params[idx], y=train_errors[idx], mode='lines',
                                   name=f'{name} Train Error'),
                        go.Scatter(x=reg_params[idx], y=validation_errors[idx], mode='lines',
                                   name=f'{name} Test Error')],
                       rows=(idx // 2) + 1, cols=(idx % 2) + 1)


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions
    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate
    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 1 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_X, train_y = X[:n_samples], y[:n_samples]
    test_X, test_y = X[n_samples:], y[n_samples:]

    # Question 2 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    ridge_params = np.linspace(0, 0.5, n_evaluations)
    lasso_params = np.linspace(0, 3, n_evaluations)

    model_classes = [LinearRegression, RidgeRegression, Lasso]
    regularization_params = [None, ridge_params, lasso_params]
    model_names = ["Linear", "Ridge", "Lasso"]

    errors = [compute_errors(model, param, train_X, train_y) for model, param in
              zip(model_classes, regularization_params)]
    train_errors, validation_errors = zip(*errors)
    fig = make_subplots(rows=1, cols=2, subplot_titles=[f"Model: {model_name}" for model_name in model_names[1:]])
    plot_errors(fig, model_names[1:], regularization_params[1:], train_errors[1:], validation_errors[1:])

    fig.update_layout(width=1200, height=600,
                      title={"x": 0.5, "text": f"Average training and validation errors for Ridge and Lasso regularizations as a function of the regularization parameter"})
    fig.write_image("Regularization_Performance_Comparison.png")

    # Question 3 - Compare best Ridge model, best Lasso model and Least Squares model
    ridge_best_param = ridge_params[np.argmin(validation_errors[1])]
    lasso_best_param = lasso_params[np.argmin(validation_errors[2])]

    ridge_model = RidgeRegression(ridge_best_param, True)
    ridge_model.fit(train_X, train_y)
    ridge_test_error = mean_square_error(test_y, ridge_model.predict(test_X))

    lasso_model = Lasso(alpha=lasso_best_param, tol=0.001)
    lasso_model.fit(train_X, train_y)
    lasso_test_error = mean_square_error(test_y, lasso_model.predict(test_X))

    least_squares_model = LinearRegression()
    least_squares_model.fit(train_X, train_y)
    least_squares_test_error = mean_square_error(test_y, least_squares_model.predict(test_X))

    print(f"Optimal regularization parameter for Ridge regression: {ridge_best_param}")
    print(f"Optimal regularization parameter for Lasso regression: {lasso_best_param}")
    print(f"Test error (Ridge regression): {ridge_test_error}")
    print(f"Test error (Lasso regression): {lasso_test_error}")
    print(f"Test error (Least Squares regression): {least_squares_test_error}")


if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()

