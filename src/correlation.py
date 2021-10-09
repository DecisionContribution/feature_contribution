import random

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from _ft_contribution import GradientBoostingRegressor


def load_concrete(return_X_y=False):
    df = pd.read_csv('data/concrete.csv')
    if return_X_y==True:
        return df.loc[:, df.columns != 'strength'],df['strength']
    else:
        return {'feature_names': df.columns}


def correlation(X, y, column: int, feature_names):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, 
                                                        random_state=28)

    reg = GradientBoostingRegressor(random_state=0, n_estimators=10)
    reg.fit(X_train, y_train)
    media, residuos, explanations = reg.decision_path(X_test)

    counter = {i: 0 for i in range(X.shape[1])}
    cols = [i[0] if i else None for i in explanations]
    for sample_residuos in residuos:
        for col, val in zip(cols, sample_residuos):
            if ~np.isnan(val) and val != 0:
                counter[col] += val / X_test.shape[0]

    df = pd.DataFrame(counter.values(), index=counter.keys(), columns=["Original"])

    ############################################################################
    print(f"\nCREATE NEW COLUMN 10 CORRELATED WITH COLUMN {column}\n")
    ############################################################################

    alfa = random.random()
    beta = random.random()
    X_train = np.column_stack((X_train, alfa * X_train[:, column] + beta))
    X_test = np.column_stack((X_test, alfa * X_test[:, column] + beta))

    levels = 5
    counter = np.zeros((X_train.shape[1], levels))

    for z in range(levels):
        reg = GradientBoostingRegressor(random_state=z, n_estimators=10)
        reg.fit(X_train, y_train)
        media, residuos, explanations = reg.decision_path(X_test)

        cols = [i[0] if i else None for i in explanations]
        for sample_residuos in residuos:
            for col, val in zip(cols, sample_residuos):
                if ~np.isnan(val) and val != 0:
                    counter[col, z] += val / X_test.shape[0]

    df = df.join(pd.DataFrame(counter), how="right")
    df.columns = ["Original"] + [f'Random #{i + 1}' for i in range(levels)]
    df.set_axis(feature_names + ['correlated'], axis=0, inplace=True)
    print(df.to_string())



def diabetes():
    X, y = load_diabetes(return_X_y=True)
    feature_names = load_diabetes()['feature_names']

    correlation(X, y, 2, feature_names)


def concrete():

    X, y = load_concrete(return_X_y=True)
    feature_names = load_concrete()['feature_names']
    X = np.array(X)
    y = np.array(y)

    correlation(X, y, 7, feature_names)


if __name__ == '__main__':

    diabetes()
    concrete()