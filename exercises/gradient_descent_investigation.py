import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type
import plotly.graph_objects as go
from sklearn import metrics

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from IMLearn.metrics import misclassification_error
from IMLearn.model_selection import cross_validate


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    objective_values, weights_list = [], []

    def callback(solver, value, weights, grad, t, eta, delta):
        objective_values.append(value)
        weights_list.append(weights.copy())

    return callback, objective_values, weights_list


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    empty_array = np.array([])
    norms = {
        "L1": L1,
        "L2": L2
    }
    minimal_loss = {norm: np.inf for norm in norms.keys()}  # Store the minimal loss for each norm

    for learning_rate in etas:
        for norm_name, norm_func in norms.items():
            callback, loss_values, parameter_values = get_gd_state_recorder_callback()
            gradient_descent = GradientDescent(FixedLR(learning_rate), callback=callback)
            optimized_weights = gradient_descent.fit(f=norm_func(init), X=empty_array, y=empty_array)

            descent_path_figure = plot_descent_path(norm_func, np.array(parameter_values),
                                                    f"Descent Path for {norm_name} Norm with Learning Rate {learning_rate}")
            descent_path_figure.write_image(f"Descent_Path_{norm_name}_Norm_Learning_Rate_{learning_rate}.png")

            convergence_rate_figure = go.Figure(
                [go.Scatter(x=np.arange(len(loss_values)), y=loss_values, mode="markers",
                            name=f"Convergence Rate for {norm_name} Norm", showlegend=True)],
                layout=go.Layout(
                    title=dict(
                        text=f"Convergence Rate of {norm_name} Norm across Gradient Descent Iterations (Learning Rate: {learning_rate})",
                        font=dict(size=10)),  # Increased font size
                    xaxis=dict(title=f"Iteration", showgrid=True, titlefont=dict(size=8)),
                    yaxis=dict(title=f"Convergence Rate of {norm_name} Norm Value", showgrid=True,
                               titlefont=dict(size=8))  # Added grid lines and increased font size
                )
            )
            convergence_rate_figure.write_image(f"Convergence_Rate_{norm_name}_Norm_Learning_Rate_{learning_rate}.png")

            minimal_loss_value = np.min(loss_values)

            # Update the minimal loss for the current norm if necessary
            if minimal_loss_value < minimal_loss[norm_name]:
                minimal_loss[norm_name] = minimal_loss_value

    for norm_name in norms.keys():
        print(f"Lowest Loss achieved when minimizing {norm_name} Norm: {minimal_loss[norm_name]}")


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    raise NotImplementedError()

    # Plot algorithm's convergence for the different values of gamma
    raise NotImplementedError()

    # Plot descent path for gamma=0.95
    raise NotImplementedError()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

    solver = GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4))
    logistic_reg = LogisticRegression(solver=solver)
    logistic_reg.fit(X_train, y_train)

    # Plotting convergence rate of logistic regression over SA heart disease data
    fpr, tpr, thresholds = metrics.roc_curve(y_train, logistic_reg.predict_proba(X_train))

    fig = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Ideal ROC Curve"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=f"ROC Curve of Logistic Regression - AUC={metrics.auc(fpr, tpr):.6f}",
                         xaxis=dict(title="False Positive Rate (FPR)"),
                         yaxis=dict(title="True Positive Rate (TPR)")))
    fig.write_image('ROC_Curve.png')

    optimal_alpha = thresholds[np.argmax(tpr - fpr)]
    test_error = misclassification_error(y_test, logistic_reg.predict_proba(X_test) >= optimal_alpha)
    print(f'Optimal ROC value is achieved with alpha = {optimal_alpha} which results in a test error of: {test_error}')

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    penalties = {
        'l1': lambda lam: LogisticRegression(solver=GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4)),
                                             penalty='l1', lam=lam),
        'l2': lambda lam: LogisticRegression(solver=GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4)),
                                             penalty='l2', lam=lam),
    }

    lam_values = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    for penalty, logistic_regression_constructor in penalties.items():
        validation_scores = [
            cross_validate(logistic_regression_constructor(lam), X_train, y_train, misclassification_error)[1] for lam
            in lam_values]
        best_lam = lam_values[np.argmin(validation_scores)]
        logistic_reg = logistic_regression_constructor(best_lam)
        logistic_reg.fit(X_train, y_train)
        loss = logistic_reg.loss(X_test, y_test)

        print(
            f'With {penalty} regularization, the optimal lambda value was {best_lam} which resulted in a loss of {loss}')


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    # compare_exponential_decay_rates() - this part was optional
    fit_logistic_regression()
