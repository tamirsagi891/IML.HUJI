from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import os

pio.templates.default = "simple_white"
preprocess_train_columns: pd.DataFrame


def filter_outliers(X: pd.DataFrame, y: pd.Series):
    """
    Filters outlier data from input feature X and target series y based
    on specified empiric conditions.
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    pd.DataFrame
        Filtered feature with outlier data removed.
    pd.Series
        Filtered target Series corresponding to the filtered feature DataFrame.
    """
    filtered_X = X.query("bedrooms >= 0 and bathrooms >= 0 and waterfront >= 0 and view >= 0 and "
                         "yr_built >= 1850 and sqft_basement <= 3500 and 1 <= grade <= 15 and floors >= 1")

    filtered_y = y.loc[filtered_X.index]
    filtered_y = filtered_y[filtered_y >= 0]
    processed_X = filtered_X.loc[filtered_y.index]
    return processed_X, filtered_y


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    train_set = y is not None

    # delete irrelevant features and replace null values with mean values:
    cleaned_data = X.drop(['id', 'date'], axis='columns')
    cleaned_data.replace('nan', np.nan, inplace=True)
    cleaned_data.fillna(cleaned_data.mean(), inplace=True)

    # validating data types:
    int_features = ['sqft_living15', 'sqft_basement', 'sqft_lot15', 'sqft_lot', 'yr_built',
                    'waterfront', 'sqft_living', 'view', 'zipcode']
    float_features = ['lat', 'long', 'bedrooms', 'bathrooms']
    cleaned_data[int_features] = cleaned_data[int_features].astype(int)
    cleaned_data[float_features] = cleaned_data[float_features].astype(float)

    # cleaning up data
    sqft_basement_mean = cleaned_data.loc[cleaned_data['sqft_basement'] > 0, 'sqft_basement'].mean()
    cleaned_data.loc[cleaned_data['sqft_basement'] == 0, 'sqft_basement'] = sqft_basement_mean
    cleaned_data['renovated_year'] = cleaned_data[['yr_built', 'yr_renovated']].max(axis=1)

    # adding new features:
    cleaned_data = pd.get_dummies(cleaned_data, prefix='zipcode_', columns=['zipcode'], dtype=int)
    cleaned_data['relative_apartment_size'] = cleaned_data['sqft_living'] / (cleaned_data['sqft_living15'] + 1)

    # differentiating between training set and testing set
    if train_set:
        return filter_outliers(cleaned_data, y)
    else:
        return cleaned_data.reindex(columns=preprocess_train_columns, fill_value=0)


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    X = X.loc[:, ~X.columns.str.contains('^zipcode_', case=False)]

    for feature in X:
        cov_mat = np.cov(X[feature], y)
        feature_pearson_correlation = cov_mat[0, 1] / np.sqrt(cov_mat[0, 0] * cov_mat[1, 1])

        fig = px.scatter(pd.DataFrame({'x': X[feature], 'y': y}), x="x", y="y",
                         title=f"{feature} vs. Response: Pearson Correlation = {feature_pearson_correlation}",
                         labels={"x": f"Feature: {feature}", "y": "Response"})
        fig.write_image(os.path.join(output_path, f'pearson_correlation_{feature}.png'), width=1000, height=700)


def fit_model(train_X, train_y, test_X, test_y):
    def percentage_generator():
        for percentage in range(10, 101):
            yield percentage

    length = 91
    loss_results = np.zeros((length, 10))

    for i, p in enumerate(percentage_generator()):
        for j in range(loss_results.shape[1]):
            sample = train_X.sample(frac=p / 100.0)
            response = train_y.loc[sample.index]
            linear_regression = LinearRegression(include_intercept=True)
            fitted_model = linear_regression.fit(sample, response)
            loss_results[i, j] = fitted_model.loss(test_X, test_y)

    loss_mean = loss_results.mean(axis=1)
    loss_standard_deviation = loss_results.std(axis=1)

    percentages = [i for i in range(10, 101)]
    lower_bound = loss_mean - 2 * loss_standard_deviation
    upper_bound = loss_mean + 2 * loss_standard_deviation
    fig = go.Figure([go.Scatter(x=percentages, y=lower_bound, fill=None, mode="lines", line=dict(color="black")),
                     go.Scatter(x=percentages, y=upper_bound, fill='tonexty', mode="lines", line=dict(color="black")),
                     go.Scatter(x=percentages, y=loss_mean, mode="markers", marker=dict(color="red"))])

    fig.update_layout(title="Mean Squared Error vs. Training Set Proportion",
                      xaxis=dict(title="Proportion of Training Data Used"),
                      yaxis=dict(title="Mean Squared Error Across Test Data"),
                      showlegend=False)

    fig.write_image("trained_model_results.png")


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    # Question 1 - split data into train and test sets
    y = df['price']
    X = df.drop(['price'], axis='columns')
    train_X, train_y, test_X, test_y = split_train_test(X, y)

    # Question 2 - Preprocessing of housing prices dataset
    train_X, train_y = preprocess_data(train_X, train_y)
    preprocess_train_columns = train_X.columns
    test_X = preprocess_data(test_X)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(train_X, train_y, "plots")

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    fit_model(train_X, train_y, test_X, test_y)
