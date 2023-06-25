import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size
    Parameters
    ----------
    n: int
        Number of samples to generate
    noise_ratio: float
        Ratio of labels to invert
    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples
    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost_model = AdaBoost(DecisionStump, n_learners)
    adaboost_model.fit(train_X, train_y)

    learner_range = range(1, n_learners + 1)
    train_errors = []
    test_errors = []

    for learner in learner_range:
        train_error = adaboost_model.partial_loss(train_X, train_y, learner)
        test_error = adaboost_model.partial_loss(test_X, test_y, learner)
        train_errors.append(train_error)
        test_errors.append(test_error)

    fig_a = go.Figure(
        data=[
            go.Scatter(x=list(learner_range), y=train_errors,
                       name=f"Training Error", mode="lines"),
            go.Scatter(x=list(learner_range), y=test_errors,
                       name=f"Test Error", mode="lines")
        ],
        layout=go.Layout(
            width=1000, height=500,
            title={"text": f"AdaBoost Misclassification Errors vs Number of Learners with Noise Ratio {noise}"},
            xaxis_title={"text": f"Number of Fitted Learners"},
            yaxis_title={"text": "Loss Error"}
        )
    )
    fig_a.write_image(f"Q1_AdaBoost_Errors_Noise_{noise}_Learners_{n_learners}.png")

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    fig_b = make_subplots(rows=2, cols=2,
                          subplot_titles=[f"Ensemble Size: {size}" for size in T])
    for i, ensemble_size in enumerate(T):
        fig_b.add_traces([
            decision_surface(
                lambda X: adaboost_model.partial_predict(X, ensemble_size),
                lims[0], lims[1],
                showscale=False),
            go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                       mode="markers", showlegend=False,
                       marker=dict(color=test_y,
                                   line=dict(color="black",
                                             width=1)))
        ], rows=(i // 2) + 1, cols=(i % 2) + 1)
    fig_b.update_layout(width=800, height=800,
                        title={"x": 0.5, "text": f"Decision Boundaries by Ensemble Size"})
    fig_b.write_image(f"Q2_Decision_Boundaries_By_Ensemble_Size_Noise_{noise}.png")

    # Question 3: Decision surface of best performing ensemble
    # TODO: read question, change graph title and compare to original answer
    min_test_loss_index = np.argmin(test_errors)
    optimal_num_learners = min_test_loss_index + 1
    accuracy = 1 - test_errors[min_test_loss_index]

    fig_c = make_subplots(rows=1, cols=1,
                          subplot_titles=[
                              f"Number of Learners: {optimal_num_learners}, Noise Level: {noise}, Accuracy: {accuracy}"])

    for i, t in enumerate([optimal_num_learners]):
        fig_c.add_traces([decision_surface(
                         lambda X: adaboost_model.partial_predict(X, t), lims[0], lims[1],
                         showscale=False),
            go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                       mode="markers", showlegend=False,
                       marker=dict(color=test_y,
                                   line=dict(color="black",
                                             width=1)))],
            rows=(i // 1) + 1, cols=(i % 1) + 1)
    fig_c.write_image(
        f"Q3_Adaboost_Decision_Surface_{optimal_num_learners}_Learners_Noise_{noise}_Accuracy_{accuracy}.png")

    # Question 4: Decision surface with weighted samples
    scaled_size_factor = 50 if noise == 0 else 10
    max_weights = np.max(adaboost_model.D_)
    scaled_weights = adaboost_model.D_ / max_weights * scaled_size_factor

    fig_d = make_subplots(rows=1, cols=1, subplot_titles=[
        f"Normalized Weights and Decision Boundary of AdaBoost Classifier on Training Set"])

    fig_d.add_traces([decision_surface(lambda input_data: adaboost_model.partial_predict(input_data, 250), lims[0],
                                       lims[1], showscale=False),
                      go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                                 marker=dict(color=train_y, size=scaled_weights,
                                             line=dict(color="black", width=1)))],
                     rows=1, cols=1)

    fig_d.write_image(f"Q4_Adaboost_Model_Weighted_Samples_{optimal_num_learners}_Learners_Noise_{noise}.png")


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)

