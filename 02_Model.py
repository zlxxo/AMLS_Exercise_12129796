from sklearn.linear_model import LinearRegression


def regression(x_train, y_train, x_test, y_test):
    reg = LinearRegression().fit(x_train, y_train)
    score_train = reg.score(x_train, y_train)
    print(f'Score on training set: {score_train}')
    score_test = reg.score(x_test, y_test)
    print(f'Score on test set: {score_test}')
    return

def classification(x_train, y_train, x_test, y_test):

    return