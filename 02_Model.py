from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import IsolationForest


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
    # regression
    reg = LogisticRegression(max_iter=1000, random_state=0).fit(x_train, y_train)
    return reg, x_train, y_train, x_test, y_test

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
    clf = svm.SVC(random_state=0)
    clf.fit(x_train, y_train)
    return clf, x_train, y_train, x_test, y_test