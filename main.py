import importlib
dataPrep = importlib.import_module("01_DataPrep")
import pandas as pd


if __name__ == '__main__':

    data = dataPrep.loadData()
    dataPrep.dataStatistics(data)

    data = dataPrep.dataNormalization(data)
    dataPrep.dataStatistics(data)

    x_train, y_train, x_val, y_val, x_test, y_test = dataPrep.splitData(data)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    print(x_val.shape)
    print(y_val.shape)
    print(type(x_train))



