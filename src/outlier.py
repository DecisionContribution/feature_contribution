from collections import defaultdict

import numpy as np
import pandas as pd
import shap
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from _ft_contribution import GradientBoostingRegressor


def load_concrete(return_X_y=False):
    df = pd.read_csv('data/concrete.csv')
    if return_X_y==True:
        return df.loc[:, df.columns != 'strength'],df['strength']
    else:
        return {'feature_names': df.columns[0:-1]}


def outlier(X, y, feature_names):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=28)
    print("DESCRIBE THE DATASET")
    print(((1000*pd.DataFrame(X).describe()).astype(int)/1000).iloc[1:,:])
    print(pd.DataFrame(y).describe())

    ############################################################################
    print("CREATE SINTHETIC OUTLIER")
    ############################################################################

    X_outlier = np.mean(X_train, axis=0).reshape((1,-1))
    X_outlier[0,1]=np.max(X_train[:,1]) + np.std(X_train[:,1])
    y_outlier = [np.max(y)+np.std(y)]

    X_train = np.vstack((X_train,X_outlier))
    y_train = np.concatenate((y_train, y_outlier))

    reg = GradientBoostingRegressor(random_state=0,
                                n_estimators=10,
                                criterion=['friedman_mse', 'squared_error', 'mae'][1],
                                max_depth=15)

    reg.fit(X_train, y_train)
    explainer = shap.Explainer(reg)


    counter_cont = defaultdict(float)
    counter_shap = defaultdict(float)

    media, residuos, explanations = reg.decision_path(X_outlier)
    shap_values = explainer(X_outlier)

    cols = [j[0] if j else None for j in explanations]
    for col, val in zip(cols, residuos[0]):
        if ~np.isnan(val) and val != 0:
            counter_cont[col] += val

    for col, val in enumerate(shap_values.values[0]):
        counter_shap[col] += val

    print("col\tcont\tshap")
    for j in range(X.shape[1]):
        print(j,"\t",
            counter_cont[j], "\t",
            counter_shap[j])

    pd.DataFrame({'contribution': counter_cont, 'shap': counter_shap})\
        .sort_index().set_axis(feature_names)\
        #.to_csv('outlier_diabetes.csv')

def diabetes():
    X, y = load_diabetes(return_X_y=True)
    feature_names = load_diabetes()['feature_names']
    outlier(X, y, feature_names)


def concrete():
    X, y = load_concrete(return_X_y=True)
    X,y = np.array(X), np.array(y)
    feature_names = load_concrete()['feature_names']
    outlier(X, y, feature_names)


if __name__ == '__main__': 

    diabetes()
    concrete()





