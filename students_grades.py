import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# loads dataset
data = pd.read_csv('student-mat.csv', sep=';')
# chooses which attributes to use from dataset
data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]
# specify which value we want to predict
predict = 'G3'
# set up an array without the value we want to predict
X = np.array(data.drop([predict], 1))
# set up an array with only the value we want to predict
y = np.array(data[predict])
# create train and test for each array (0.1 = 10% of data)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
'''
# loops to find a model with a high accuracy.
# once a good model is found, this part will be commented out
# uncomment to find a model with a different accuracy
best = 0
for _ in range(30):
    # create train and test for each array (0.1 = 10% of data)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
    # linear regression model
    linear = linear_model.LinearRegression()
    # find a best-fit line
    linear.fit(x_train, y_train)
    # check accuracy of model
    acc = linear.score(x_test, y_test)
    print(acc)
    if acc > best:
        best = acc
        # save model so testing is not needed anymore
        # save model "linear" into "f"
        with open('studentmodel.pickle', 'wb') as f:
            pickle.dump(linear, f)
'''
# read pickle file
pickle_in = open('studentmodel.pickle', 'rb')
# load this into linear model
linear = pickle.load(pickle_in)

# check line coefficients and intercept
print('Coefficients:\n', linear.coef_)
print('Intercept:\n', linear.intercept_)
# predict
predictions = linear.predict(x_test)
# loops through the array x_test and shows:
# the predicted and the actual final grade
for x in range(len(predictions)):
    print('Predicted: ' + str(predictions[x]) + ' - Actual: ' + str(y_test[x]))

# PLOTS
p = 'G1'
# make grid look better
style.use('ggplot')
# scatter plot
pyplot.scatter(data[p], data['G3'])
# create and show axis labels
pyplot.xlabel(p)
pyplot.ylabel('Final Grade')
pyplot.show()





