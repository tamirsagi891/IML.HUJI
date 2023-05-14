import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """

    # read the data and dropping unnecessary lines (duplicated and nulls)
    temp_data = pd.read_csv(filename, parse_dates=['Date'])
    temp_data = temp_data.dropna()
    temp_data = temp_data.drop_duplicates()
    temp_data = temp_data[temp_data.Temp > 0]

    # editing the data and adding features
    temp_data['DayOfYear'] = temp_data['Date'].dt.dayofyear
    temp_data['Year'] = temp_data['Year'].astype(str)
    temp_data['DM'] = temp_data['Date'].dt.strftime('%d-%m')
    return temp_data


def plot_country_data(data: pd.DataFrame, country: str) -> pd.DataFrame:
    country_data = data[data.Country == country]

    fig = px.scatter(country_data, x='DayOfYear', y='Temp', color='Year',
                     title=f'{country} daily temperatures average per year')
    fig.write_image(f'{country}_daily_temperatures.png')

    monthly_std = country_data.groupby(['Month'], as_index=False).agg(std=('Temp', 'std'))
    plot = px.bar(monthly_std, title='STD of average monthly temperatures in Israel over years', x='Month', y='std')
    plot.write_image(f'average_temp_per_month_for_{country}.png')

    return country_data


def plot_differences(data: pd.DataFrame):
    country_data = data.groupby(['Country', 'Month'], as_index=False)
    fig = px.line(country_data.mean(), x='Month', y='Temp', color='Country', error_y=(country_data.std()['Temp']))
    fig.update_layout(title='monthly avg temperature per country',
                      xaxis_title='month', yaxis_title='temperature avg')
    fig.write_image("average_monthly_temp_per_country.png")


def evaluate_model_on_all_other_countries(data: pd.DataFrame, israel_data: pd.DataFrame, degree: int) -> None:
    fit = PolynomialFitting(k=degree).fit(israel_data.DayOfYear.to_numpy(), israel_data.Temp.to_numpy())
    countries = data.Country.unique()
    other_countries = [country for country in countries if country != 'Israel']
    country_errors = [{"country": country,
                       "error": round(fit.loss(data[data.Country == country].DayOfYear,
                                               data[data.Country == country].Temp), 2)} for country in other_countries]
    errors_df = pd.DataFrame(country_errors)
    err_fig = px.bar(errors_df, x="country", y="error", text="error",
                            color="country",
                            title="Comparative Loss of 5th-Degree Polynomial Model Fitted to Israel vs. Other Countries")
    err_fig.write_image("model_error_over_other_countries.png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_df = plot_country_data(df, "Israel")

    # Question 3 - Exploring differences between countries
    plot_differences(df)

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(israel_df.DayOfYear, israel_df.Temp)
    k_vals = [k for k in range(1, 11)]
    loss = np.zeros(len(k_vals), dtype=float)
    for i, k in enumerate(k_vals):
        poly_model = PolynomialFitting(k=k).fit(train_X.to_numpy(), train_y.to_numpy())
        loss[i] = np.round(poly_model.loss(test_X.to_numpy(), test_y.to_numpy()), 2)

    loss = pd.DataFrame(dict(k=k_vals, loss=loss))
    fig = px.bar(loss, x="k", y="loss", text="loss",
                 title=f'Error in Israel Polynomial Fit Model for Varying Polynomial Degrees (k)')
    fig.write_image("poly_model_for_israel_different_k_vals", format='png')
    print(loss)

    # Question 5 - Evaluating fitted model on different countries
    evaluate_model_on_all_other_countries(df, israel_df, 5)
