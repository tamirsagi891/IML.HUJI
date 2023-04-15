from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu = 10
    sig = 1
    samples_set_size = 1000
    X = np.random.normal(loc=mu, scale=sig, size=samples_set_size)
    fit = UnivariateGaussian().fit(X)
    predictions = (fit.mu_, fit.var_)
    print(predictions)

    # Question 2 - Empirically showing sample mean is consistent
    sample_lens = [n * 10 for n in range(1, int(len(X) / 10))]
    mu_hat = [(np.abs(mu - UnivariateGaussian().fit(X[:sample_len]).mu_)) for sample_len in sample_lens]

    go.Figure(go.Scatter(x=list(range(len(mu_hat))), y=mu_hat, mode=f'markers', marker=dict(color=f'black')),
              layout=dict(title=f'Deviation of Sample Mean Estimation According to Sample Set Size',
                          xaxis_title=f'Sample Size',
                          yaxis_title=f'Sample Mean Estimator')) \
        .write_image(f'deviation_of_sample_mean.png')

    # Question 3 - Plotting Empirical PDF of fitted model
    go.Figure(go.Scatter(x=X, y=fit.pdf(X), mode="markers", marker=dict(color="black")),
              layout=dict(title=f"Sample Probability as Function of it's Values",
                          xaxis_title=f'Sample Value',
                          yaxis_title=f'Sample Probability')) \
        .write_image(f'probability_density_function.png')


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = [0, 0, 4, 0]
    cov = np.array([[1, 0.2, 0, 0.5],
                    [0.2, 2, 0, 0],
                    [0, 0, 1, 0],
                    [0.5, 0, 0, 1]])
    samples_set_size = 1000
    X = np.random.multivariate_normal(mean=mu, cov=cov, size=samples_set_size)
    fit = MultivariateGaussian().fit(X)
    print(fit.mu_)
    print(fit.cov_)

    # Question 5 - Likelihood evaluation
    linspace_interval: int = 200
    value_pool = np.linspace(-10, 10, linspace_interval)
    log_likelihood_values = [MultivariateGaussian.log_likelihood(np.array([f1, 0, f3, 0]), cov, X) for f1 in value_pool
                             for f3 in value_pool]
    log_likelihood_values = np.reshape(log_likelihood_values, (linspace_interval, linspace_interval))

    go.Figure(go.Heatmap(x=value_pool, y=value_pool, z=log_likelihood_values),
              layout=dict(title=dict(text=f'Log Likelihood of Multivariate Gaussian Based on Expectation Features 1&3'
                                          f' With a Defined Covariance Matrix', font=dict(size=11)),
                          xaxis_title=f'Feature 3', yaxis_title=f'Feature 1')).write_image(f'likelihood_heatmap.png')

    # Question 6 - Maximum likelihood
    max_log_likelihood_inds = np.unravel_index(np.argmax(log_likelihood_values), log_likelihood_values.shape)
    max_log_likelihood_vals = (round(value_pool[max_log_likelihood_inds[0]], 3),
                               round(value_pool[max_log_likelihood_inds[1]], 3))
    print(max_log_likelihood_vals)


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
