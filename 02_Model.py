from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error


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