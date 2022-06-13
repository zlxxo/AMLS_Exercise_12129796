from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from imblearn.over_sampling import SMOTE

labels = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
          'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

def cleanData(data):
    columns = np.array(data.columns)
    size = columns.shape[0]
    y = data[[columns[size - 2], columns[size - 1]]]
    x = data.drop([columns[size - 2], columns[size - 1]], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)

    # removing correlated columns
    correlation_matrix = data.corr()
    correlated_columns = []

    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.75:
                column = correlation_matrix.columns[i]
                correlated_columns.append(column)

    print(f'Correlated columns: {correlated_columns}')

    x_train = x_train.drop(labels=correlated_columns, axis=1)
    x_test = x_test.drop(labels=correlated_columns, axis=1)

    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()
    x_test = x_test.to_numpy()
    y_test = y_test.to_numpy()

    return x_train, y_train, x_test, y_test


def oversampleData(x, y):
    print(f'Number of training samples: {x.shape[0]}')
    oversample = SMOTE(k_neighbors=2, random_state=0)
    x, y = oversample.fit_resample(x, y)
    print(f'Number of training samples after oversampling: {x.shape[0]}')
    return x, y


def addNonlinearities(x_train, y_train, x_test, y_test):

    # plotting combinations of variables with corresponding targets to see if there
    # non-linearities that are visible to eye
    un_val = np.unique(y_train)
    colors = np.array(['yellow', 'pink', 'red', 'blue', 'grey', 'black', 'brown'])
    w = x_test.shape[1]
    for row in range(w):
        for col in range(row):
            plt.figure(row * 11 + col)
            for i in range(4):
                for j in range(2):
                    if i * 2 + j >= un_val.shape[0]:
                        break
                    val = un_val[i * 2 + j]
                    color = colors[i * 2 + j]
                    plt.scatter(x_test[:, row][y_test == val], x_test[:, col][y_test == val], c=color)
            plt.title(f'Variables {row} and {col}')
            plt.show()

    #poly = PolynomialFeatures(2)
    #x_poly_train = np.array(x_train)
    #poly.fit_transform(x_poly_train)
    #x_poly_test = np.array(x_test)
    #poly.fit_transform(x_poly_test)
    return x_train, x_test

def regression(x_train, y_train, x_test, y_test):
    # identify outliers
    iso = IsolationForest(contamination=0.2)
    yhat = iso.fit_predict(x_train)
    mask = yhat != -1
    x_train, y_train = x_train[mask, :], y_train[mask]
    # scaling data
    scaler = MinMaxScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    # non-linearities
    x_train, x_test = addNonlinearities(x_train, y_train, x_test, y_test)
    # regression
    reg = LogisticRegression(max_iter=1000).fit(x_train, y_train)
    score_train = reg.score(x_train, y_train)
    print(f'Score on training set: {score_train}')
    y_train_pred = reg.predict(x_train)
    mae_train = mean_absolute_error(y_train_pred, y_train)
    print(f'Mean absolute error on training set: {mae_train}')
    score_test = reg.score(x_test, y_test)
    print(f'Score on test set: {score_test}')
    y_test_pred = reg.predict(x_test)
    mae_test = mean_absolute_error(y_test_pred, y_test)
    print(f'Mean absolute error on test set: {mae_test}')
    return

def classification(x_train, y_train, x_test, y_test):
    # identify outliers
    iso = IsolationForest(contamination=0.2)
    yhat = iso.fit_predict(x_train)
    mask = yhat != -1
    x_train, y_train = x_train[mask, :], y_train[mask]
    #scaling data
    scaler = MinMaxScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    # classification
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    score_train = clf.score(x_train, y_train)
    print(f'Score on training set: {score_train}')
    y_train_pred = clf.predict(x_train)
    mae_train = mean_absolute_error(y_train_pred, y_train)
    print(f'Mean absolute error on training set: {mae_train}')
    score_test = clf.score(x_test, y_test)
    print(f'Score on test set: {score_test}')
    y_test_pred = clf.predict(x_test)
    mae_test = mean_absolute_error(y_test_pred, y_test)
    print(f'Mean absolute error on test set: {mae_test}')
    return