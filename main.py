import importlib
dataPrep = importlib.import_module("01_DataPrep")
Model = importlib.import_module("02_Model")


if __name__ == '__main__':

    data = dataPrep.loadData()
    #dataPrep.dataStatistics(data)

    data = dataPrep.dataNormalization(data)
    #dataPrep.dataStatistics(data)

    x_train, y_train, x_test, y_test, _, _ = dataPrep.splitData(data, validation_split=0.)

    Model.regression(x_train, y_train[:, 0], x_test, y_test[:, 0])




