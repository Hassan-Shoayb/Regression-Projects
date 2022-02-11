# Linear Model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Data Analysis
import numpy as np
import pandas as pd
from pandas import DataFrame

# Gather Data
life_data = pd.read_csv("Life Data.csv")
target = np.log(life_data.LIFE_EXP)

# Dropping some features to due to insignificance and Multocollinearity
features = life_data.drop(
    ["LIFE_EXP", "HEPT-B", "BMI", "POLIO", "POPLN", "THIN_1_19", "SCH"], axis=1)

YEAR = 0
ADLT_MOR = 1
AIDS = 5
life_stats = np.ndarray(shape=(1, 8))
life_stats = features.mean().values.reshape(1, 8)

# Running Regression
regr = LinearRegression()
regr.fit(features, target)

# Calculatin  the MSE and RMSE
MSE = mean_squared_error(target, regr.predict(features))
RMSE = np.sqrt(mean_squared_error(target, regr.predict(features)))

def life_estimate(year, adult_mortality, aids):

    # Configure Life Expectancy
    life_stats[0][YEAR] = year
    life_stats[0][ADLT_MOR] = adult_mortality
    life_stats[0][AIDS] = aids

    # Make Predictions
    life_exp = regr.predict(life_stats)[0]

    # Calculating The Upper and Lower-bound of Life Estimates
    upper_bound = life_exp + 2*RMSE
    lower_bound = life_exp - 2*RMSE

    return np.e**life_exp, np.e**upper_bound, np.e**lower_bound


def life_expectancy(year, adult_mortality, percent_aids):
    '''
        Estimate Life Expectancy 

        KEYWORD ARGUMESNT
        Year -- Year at which you will find the Life Expectancy.
        adult_mortality -- Number of Adult deaths at that period of year.
        percent_aids -- Percentage of people Infected by HIV/AIDS at that period of year.

    '''

    if percent_aids > 100:
        print('Aids Percentage Execeeded !')
        return

    life_exp, higher_exp, lower_exp = life_estimate(
        year, adult_mortality, percent_aids)

    print(f"Life Expectancy: {round(life_exp)} Years")
    print(f"Higher Level Expectancy: {round(higher_exp)} Years")
    print(f"Lower Level Expectancy: {round(lower_exp)} Years")


life_expectancy(2020, 200, 30)
