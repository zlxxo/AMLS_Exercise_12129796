import importlib
dataPrep = importlib.import_module("01_DataPrep")
Model = importlib.import_module("02_Model")
Tuning = importlib.import_module("03_Tuning")
Debug = importlib.import_module("04_Debug")


if __name__ == '__main__':

    print('----- Load data and check for null and NaN values ------')
    data = dataPrep.loadData()

    print('----- Check statistics of data variables ------')
    dataPrep.dataStatistics(data)

    x_train, y_train, x_test, y_test, _, _ = dataPrep.splitData(data, validation_split=0.)

    print('----- Regression ------')
    model, x_tr, y_tr, x_te, y_te = Model.regression(x_train, y_train[:, 0], x_test, y_test[:, 0])
    Debug.score(model, x_tr, y_tr, x_te, y_te)
    Debug.residuals(model, x_tr, y_tr, x_te, y_te)

    print('----- Classification ------')
    model, x_tr, y_tr, x_te, y_te = Model.classification(x_train, y_train[:, 1], x_test, y_test[:, 1])
    Debug.score(model, x_tr, y_tr, x_te, y_te)
    Debug.confusionMatrix(model, x_tr, y_tr, x_te, y_te, 'Confusion matrix of classification model on training data',
                          'Confusion matrix of classification model on test data')

    print('----- Clean data ------')
    x_train, y_train, x_test, y_test = Tuning.cleanData(data)

    print('----- Tuned Regression ------')
    model, x_tr, y_tr, x_te, y_te = Tuning.regression(x_train, y_train[:, 0], x_test, y_test[:, 0])
    Debug.score(model, x_tr, y_tr, x_te, y_te)
    Debug.residuals(model, x_tr, y_tr, x_te, y_te)

    print('----- Tuned Classification ------')
    model, x_tr, y_tr, x_te, y_te = Tuning.classification(x_train, y_train[:, 1], x_test, y_test[:, 1])
    Debug.score(model, x_tr, y_tr, x_te, y_te)
    Debug.confusionMatrix(model, x_tr, y_tr, x_te, y_te, 'Confusion matrix of tuned classification model on training data',
                          'Confusion matrix of tuned classification model on test data')




