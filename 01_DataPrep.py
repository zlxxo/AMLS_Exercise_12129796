import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def loadData():
    red = pd.read_csv('data/winequality-red.csv', sep=';')
    h, w = red.shape
    labels = np.zeros(h)
    red.insert(w, 'wine_type', labels)
    red = red.astype({'wine_type': 'int32'})
    white = pd.read_csv('data/winequality-white.csv', sep=';')
    h, w = white.shape
    labels = np.ones(h)
    white.insert(w, 'wine_type', labels)
    white = white.astype({'wine_type': 'int32'})
    frames = [red, white]
    data = pd.concat(frames, ignore_index=True)
    # check if there are null values
    null_values = data.isnull().sum()
    print('Null values in dataset:')
    print(null_values)
    nan_values = pd.isna(data).sum()
    print('NaN values in dataset')
    print(nan_values)
    return data


def dataStatistics(data):
    statistics = data.describe()
    print(statistics.to_string())
    data.plot(subplots=True, layout=(5,3), figsize=(10, 10))
    plt.title('Dataset variables')
    plt.show()
    data.hist(bins=20,figsize=(10,10))
    plt.title('Variable histograms')
    plt.show()
    return


def dataNormalization(data):
    # min-max normalization
    normalized_data = (data - data.min()) / (data.max() - data.min())
    return normalized_data


def splitData(data, validation_split=0.2):
    columns = np.array(data.columns)
    size = columns.shape[0]
    y = data[[columns[size - 2], columns[size - 1]]]
    x = data.drop([columns[size - 2], columns[size - 1]],axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_val = []
    y_val = []
    if(validation_split > 0.):
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_split)

    return x_train, y_train, x_val, y_val, x_test, y_test