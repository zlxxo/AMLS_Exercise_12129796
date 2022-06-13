import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def confusionMatrix(model, x_train, y_train, x_test, y_test, title1, title2):
    y_pred = model.predict(x_train)
    conf_matrix = confusion_matrix(y_train, y_pred)
    conf_data = pd.DataFrame(conf_matrix)
    sn.heatmap(conf_data, annot=True)
    plt.title(title1)
    plt.show()

    y_pred = model.predict(x_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_data = pd.DataFrame(conf_matrix)
    sn.heatmap(conf_data, annot=True)
    plt.title(title2)
    plt.show()
    return

def score(model, x_train, y_train, x_test, y_test):
    score_train = model.score(x_train, y_train)
    print(f'Score on training set: {score_train}')
    score_test = model.score(x_test, y_test)
    print(f'Score on test set: {score_test}')
    return


def residuals(model, x_train, y_train, x_test, y_test):
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    print(f'Mean absolute error on training set: {mae_train}')
    mae_test = mean_absolute_error(y_test, y_pred_test)
    print(f'Mean absolute error on test set: {mae_test}')
    mse_train = mean_squared_error(y_train, y_pred_train)
    print(f'Mean squared error on training set: {mse_train}')
    mse_test = mean_squared_error(y_test, y_pred_test)
    print(f'Mean squared error on test set: {mse_test}')
    return

def coefficients(model):
    print(f'Coefficients of the model: {model.coef_}')