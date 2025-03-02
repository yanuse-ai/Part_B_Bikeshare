from typing import List
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class WeekdayImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weekday' column by extracting dayname from 'dteday' column """

    def __init__(self, variables: str):
        if not isinstance(variables, str):
            raise ValueError("variables should be a list")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        self.fill_value=X[self.variables].mode()[0]
        return self

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:

        df = dataframe.copy()
        wkday_null_idx = df[df[self.variables].isnull() == True].index
        # print(len(wkday_null_idx))
        # df['dteday'] = pd.to_datetime(df['dteday'])
        df.loc[wkday_null_idx, self.variables] = df.loc[wkday_null_idx, 'dteday'].dt.day_name().apply(lambda x: x[:3])

        return df

class WeathersitImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weathersit' column by replacing them with the most frequent category value """

    def __init__(self, variables: str):
        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # X[self.variables].fillna('Clear', inplace=True)
        # 'df.method({col: value}, inplace=True)'
        X.fillna({self.variables: 'Clear'}, inplace=True)
        return X    


class Mapper(BaseEstimator, TransformerMixin):
    """
    Ordinal categorical variable mapper:
    Treat column as Ordinal categorical variable, and assign values accordingly
    """

    def __init__(self, variables: str, mappings: dict):

        if not isinstance(variables, str):
            raise ValueError("variables should be a str")

        self.variables = variables
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        #for feature in self.variables:
        X[self.variables] = X[self.variables].map(self.mappings).astype(int)

        return X
    
class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values:
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self, variables: str):
        if isinstance(variables, str):
           self.variables = [variables] # if it's a single variable transform to list
        elif not isinstance(variables, list):
            raise ValueError("variables should be a list or string")
        else:
            self.variables=variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for var in self.variables:  # loop over all numerical variables
            q1 = X.describe()[var].loc['25%']
            q3 = X.describe()[var].loc['75%']
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            for i in X.index:
                if X.loc[i,var] > upper_bound:
                    X.loc[i,var]= upper_bound
                if X.loc[i,var] < lower_bound:
                    X.loc[i,var]= lower_bound

        return X


class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode weekday column """

    def __init__(self, variables: str):
        self.variables = variables
        self.encoder = OneHotEncoder(sparse_output=False)

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.encoder.fit(X[[self.variables]])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        encoded_weekday = self.encoder.transform(X[['weekday']])
        enc_wkday_features = self.encoder.get_feature_names_out(['weekday'])
        X[enc_wkday_features] = encoded_weekday
        # Drop original weekday column after one-hot encoding
        X.drop(columns=[self.variables], inplace=True)
        return X


class ColumnDropper(BaseEstimator, TransformerMixin):
    """ Drop specified columns """
    def __init__(self, variables: list):
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")
        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X.drop(labels = self.variables, axis = 1, inplace = True)
        return X

yr_mapping = Mapper('yr',{2011: 0, 2012: 1})
mnth_mapping = Mapper('mnth',{'January': 0, 'February': 1, 'December': 2, 'March': 3, 'November': 4, 'April': 5,
                'October': 6, 'May': 7, 'September': 8, 'June': 9, 'July': 10, 'August': 11})
season_mapping = Mapper('season',{'spring': 0, 'winter': 1, 'summer': 2, 'fall': 3})
weather_mapping = Mapper('weathersit',{'Heavy Rain': 0, 'Light Rain': 1, 'Mist': 2, 'Clear': 3})
holiday_mapping = Mapper('holiday',{'Yes': 0, 'No': 1})
workingday_mapping = Mapper('workingday',{'No': 0, 'Yes': 1})
hour_mapping = Mapper('hr',{'4am': 0, '3am': 1, '5am': 2, '2am': 3, '1am': 4, '12am': 5, '6am': 6, '11pm': 7, '10pm': 8,
                '10am': 9, '9pm': 10, '11am': 11, '7am': 12, '9am': 13, '8pm': 14, '2pm': 15, '1pm': 16,
                '12pm': 17, '3pm': 18, '4pm': 19, '7pm': 20, '8am': 21, '6pm': 22, '5pm': 23})    