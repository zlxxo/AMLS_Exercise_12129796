from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import IsolationForest

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



def regression(x_train, y_train, x_test, y_test):
    # remove outliers
    x_train, y_train = removeOutliers(x_train, y_train)
    # scaling data
    x_train, x_test = scaleData(x_train, x_test)
    # regression
    reg = LogisticRegression(max_iter=1000, random_state=0).fit(x_train, y_train)
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