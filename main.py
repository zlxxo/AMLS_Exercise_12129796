import importlib
dataPrep = importlib.import_module("01_DataPrep")
Model = importlib.import_module("02_Model")
Tuning = importlib.import_module("03_Tuning")


if __name__ == '__main__':

    print('----- Load data and check for null and NaN values ------')
    data = dataPrep.loadData()

    print('----- Check statistics of data variables ------')
    #dataPrep.dataStatistics(data)

    x_train, y_train, x_test, y_test, _, _ = dataPrep.splitData(data, validation_split=0.)

    print('----- Regression ------')
    Model.regression(x_train, y_train[:, 0], x_test, y_test[:, 0])

    print('----- Classification ------')
    Model.classification(x_train, y_train[:, 1], x_test, y_test[:, 1])

    print('----- Clean data ------')
    x_train, y_train, x_test, y_test = Tuning.cleanData(data)

    print('----- Tuned Regression ------')
    Tuning.regression(x_train, y_train[:, 0], x_test, y_test[:, 0])

    print('----- Tuned Classification ------')
    Tuning.classification(x_train, y_train[:, 1], x_test, y_test[:, 1])




