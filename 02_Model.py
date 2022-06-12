from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import IsolationForest


def regression(x_train, y_train, x_test, y_test):
    # identify outliers
    iso = IsolationForest(contamination=0.1)
    yhat = iso.fit_predict(x_train)
    mask = yhat != -1
    x_train, y_train = x_train[mask, :], y_train[mask]
    # scaling data
    scaler = MinMaxScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    # regression
    reg = LogisticRegression(max_iter=10000, penalty='l1', solver='saga').fit(x_train, y_train)
    score_train = reg.score(x_train, y_train)
    print(f'Score on training set: {score_train}')
    score_test = reg.score(x_test, y_test)
    print(f'Score on test set: {score_test}')
    return

def classification(x_train, y_train, x_test, y_test):
    # identify outliers
    iso = IsolationForest(contamination=0.1)
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
    score_test = clf.score(x_test, y_test)
    print(f'Score on test set: {score_test}')
    return