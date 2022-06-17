------------------------------------------------
---------------- Documentation -----------------
------------------------------------------------


------------ How to run the code? --------------

Evrything is writen in python and the tasks are divided in individual files. However in each file
contains methods that do spacific tasks in order to be able to reuse code in other tasks.
Methods are then called in the main.py file to produce results. By running main.py you get the results.

To run the code following libraries have to be installed:
- sklearn
- imblearn
- matplotlib
- pandas
- seaborn



------------------ Task 1.1 --------------------

In 01_DataPrep.py file we first load data in loadDara() method. Both .cvs files are read and combined
and the column is added to label type of wine (white or red) for classification.
We also check if there are samples with undefined values (NaN or Null). But in this set there are no such samples.

Then in dataStatistics() method we first get meaningful statistical data such as mean, standard deviation, max and min values, etc.
This tells us more about variables and how are they distributed.
But we then plot the data, its histograms and distribution. This gives us a visual overview of the data.
In the first plot we just plot the variables in order that they are sampled. We do that to see what values are present and
try to see if there are values that differ a lot from the rest.
In the second plot we plot histograms. In histograms we group data in bins and get an overview of how the bins are distributed.
In the third plot we plot variable distributions. There we can see what values are present in the variable and how often
are they sampled in the set. We can already notice that there are some variables where the sampled data is in a very small range
and seems constant in a way.

In the splitData() method we split the data in 3 sets:
- tarining set
- validation set
- test set.
In this method we split the original data in 80:20 ratio to training and test sets respectively. And if necessary
we take the part of the training set as a validation set.



------------------ Task 1.2 --------------------

In 02_Model.py file there are two methods, regression() and classification() respectively to the tasks that we are solving.
The other two methods are for removing outliers and scaling data.

In the first plot of the previous task we noticed that are some data points that differ from the rest of the data, we try
to remove those samples by removing outliers. We do that in the removeOutliers() method and what made a bit of change on
the performance was to remove 0.2 of the contamination in the set.

In  the scaleData() method we perform min-max normalization of the features. After this all the data is in range [0, 1].
We also tried performing standardization after which mean of the data is 0 and standard deviation is 1, but it does not
affect the performance of the models

In regression() method we first remove outliers and perform normalization. We use LogisticRegression model from sklearn
since it produced the best performance. We also tried using different models such as:
- LinearRegression from sklearn
- KNeighborsRegressor from sklearn
- RandomForestRegressor from sklearn
- DecisionTreeRegressor from sklearn
- MLPRegressor from sklearn with different number of hidden layers and number of units in those layers
- keras Sequential model with dense layers


In the classification() method we first remove outliers and perform normalization. We use a support-vector model
for classification (SVC) from sklearn and in the beginning it performs well with the given dataset.