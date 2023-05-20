from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:

        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def callback(p: Perceptron, xn: np.ndarray, yn: int):
            losses.append(p._loss(X, y))

        perceptron = Perceptron(callback=callback)
        perceptron.fit(X, y)

        # Plot figure of loss as function of fitting iteration
        fig = go.Figure(
            data=go.Scatter(
                x=np.arange(len(losses)),
                y=losses,
                mode="lines",
                marker=dict(color="black")
            ),
            layout=go.Layout(
                title={"text": f"Perceptron Training Loss Progression: Done on {n} Dataset"},
                xaxis=dict(title="Iterations", linecolor='black', mirror=True),
                yaxis=dict(title="Training Loss", linecolor='black', mirror=True),
            )
        )

        fig.write_image(f"perceptron_training_loss_{n}.png")


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit models and predict over training set
        lda_model = LDA().fit(X, y)
        lda_predictions = lda_model.predict(X)

        naive_bayes_model = GaussianNaiveBayes().fit(X, y)
        naive_bayes_predictions = naive_bayes_model.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy

        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            f"Gaussian Naive Bayes [Accuracy = {round(accuracy(naive_bayes_predictions, y), 3):.3f}]",
            f"Linear Discriminant Analysis [Accuracy = {round(accuracy(lda_predictions, y), 3):.3f}]"))

        # Add traces for data-points setting symbols and colors
        marker_params = dict(symbol=class_symbols[y], colorscale=class_colors(3))
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", marker=dict(color=naive_bayes_predictions,
                                                                                   **marker_params)), row=1, col=1)
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", marker=dict(color=lda_predictions,
                                                                                   **marker_params)), row=1, col=2)

        # Add `X` dots specifying fitted Gaussians' means
        mean_marker_params = dict(symbol="x", color="black", size=12)
        fig.add_trace(go.Scatter(x=naive_bayes_model.mu_[:, 0], y=naive_bayes_model.mu_[:, 1], mode="markers",
                                 marker=mean_marker_params), row=1, col=1)
        fig.add_trace(go.Scatter(x=lda_model.mu_[:, 0], y=lda_model.mu_[:, 1], mode="markers",
                                 marker=mean_marker_params), row=1, col=2)

        # Add ellipses depicting the covariances of the fitted Gaussians
        naive_bayes_ellipses = [get_ellipse(mu, np.diag(var)) for mu, var in zip(naive_bayes_model.mu_,
                                                                                 naive_bayes_model.vars_)]
        lda_ellipses = [get_ellipse(mu, lda_model.cov_) for mu in lda_model.mu_]
        fig.add_traces(naive_bayes_ellipses,
                       rows=[1] * len(naive_bayes_model.mu_),
                       cols=[1] * len(naive_bayes_model.mu_))
        fig.add_traces(lda_ellipses,
                       rows=[1] * len(lda_model.mu_),
                       cols=[2] * len(lda_model.mu_))

        dataset_name = f.split('.')[0] if f'.' in f else f
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_layout(title_text=f"Performance of Naive Bayes vs LDA on [{dataset_name}] Dataset", title_x=0.5,
                          width=1000, height=500, showlegend=False)
        fig.write_image(f"comparison_plot_using_{dataset_name}_dataset.png")


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
