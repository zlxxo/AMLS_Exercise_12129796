from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

labels = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
          'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']


def removeOutliers(x_train, y_train):
    # identify outliers
    iso = IsolationForest(contamination=0.2)
    y = iso.fit_predict(x_train)
    mask = y != -1
    return x_train[mask, :], y_train[mask]

def scaleData(x_train, x_test):
    scaler = MinMaxScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test


def removeCorrelatedDataAndSplit(data):
    columns = np.array(data.columns)
    size = columns.shape[0]
    y = data[[columns[size - 2], columns[size - 1]]]
    x = data.drop([columns[size - 2], columns[size - 1]], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)

    # removing correlated columns
    correlation_matrix = data.corr(method='spearman')
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

    poly = PolynomialFeatures(2)
    x_poly_train = np.array(x_train)
    poly.fit_transform(x_poly_train)
    x_poly_test = np.array(x_test)
    poly.fit_transform(x_poly_test)
    return x_train, x_test

def splitTobBins(x_train, x_test, n_bins):
    mins = np.min(x_train, axis=0)
    maxs = np.max(x_train, axis=0)

    h, w = x_train.shape
    x_train_bins = np.zeros(shape=(h, w))
    for i in range(n_bins):
        for m in range(h):
            for n in range(w):
                if x_train[m, n] >= ((maxs[n] - mins[n]) / n_bins) * i and x_train[m, n] < ((maxs[n] - mins[n]) / n_bins) * (i + 1.):
                    x_train_bins[m, n] = i / n_bins

    h, w = x_test.shape
    x_test_bins = np.zeros(shape=(h, w))
    for i in range(n_bins):
        for m in range(h):
            for n in range(w):
                if x_test[m, n] >= ((maxs[n] - mins[n]) / n_bins) * i and x_test[m, n] < (
                        (maxs[n] - mins[n]) / n_bins) * (i + 1.):
                    x_test_bins[m, n] = i / n_bins

    return x_train_bins, x_test_bins

def regression(x_train, y_train, x_test, y_test):
    # remove outliers
    x_train, y_train = removeOutliers(x_train, y_train)
    # scaling data
    x_train, x_test = scaleData(x_train, x_test)
    # oversampling training set because uneven distribution of targets
    #x_train, y_train = oversampleData(x_train, y_train)
    # binning data
    #x_train, x_test = splitTobBins(x_train, x_test, 10)
    # add non-linearities
    #x_train, x_test = addNonlinearities(x_train, y_train, x_test, y_test)
    # regression
    reg = LogisticRegression(max_iter=10000, random_state=0).fit(x_train, y_train)
    return reg, x_train, y_train, x_test, y_test

def classification(x_train, y_train, x_test, y_test):
    # remove outliers
    x_train, y_train = removeOutliers(x_train, y_train)
    # scaling data
    x_train, x_test = scaleData(x_train, x_test)
    # classification
    clf = svm.SVC(random_state=0)
    clf.fit(x_train, y_train)
    return clf, x_train, y_train, x_test, y_test