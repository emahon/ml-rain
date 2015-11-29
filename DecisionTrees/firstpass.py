"""
First attempt at using boosted decision trees to predict
Instructions on installing xgboost:
http://xgboost.readthedocs.org/en/latest/python/python_intro.html
"""

import csv
import gc
import numpy
import numpy.random as rand
import xgboost
import to_output

#read the csv file into a numpy ndarray
inputreader = csv.reader(open("../average.csv",'r'), delimiter=",")
data = []
for row in inputreader:
    #convert strings to floats
    converted = []
    for j in row:
        if (len(j) > 0):
            converted.append(float(j))
        else:
            converted.append(float("nan"))
    data.append(converted)

data = numpy.array(data)
print data.shape
#split into test and train
rand.shuffle(data)
split = numpy.floor(.8*data.shape[0])
Xtrain = data[:split,:-1]
ytrain = data[:split,-1]
#Xtest = data[split:,:-1]
#ytest = data[split:,-1]
#train the model
dtrain = xgboost.DMatrix(Xtrain,label=ytrain,missing=float("nan"))
params = dict()
model = xgboost.train(params,dtrain)
#dtest = xgboost.DMatrix(Xtest, missing=float("nan"))
#run a prediction to test
#pred = model.predict(dtest)
#vals = numpy.abs(pred - ytest)/ytest
#print numpy.mean(vals)
#clean up old data to free memory
data = []
Xtrain = []
ytrain = []
#Xtest = []
#ytest = []
dtrain = []
#dtest = []
gc.collect()
print "cleaning memory...."
#now run on the actual testing data for kaggle
testreader = csv.reader(open("../avtest.csv",'r'), delimiter=",")
test = []
i = 0
for row in testreader:
    i += 1
    #convert strings to floats
    converted = []
    #remove old converted every 5000 cycles
    if ((i % 5000.0) == 0):
        print "clean up"+str(i)
        gc.collect()
    for j in row:
        if (len(j) > 0):
            converted.append(float(j))
        else:
            converted.append(float("nan"))
    test.append(converted)

print "done looping"
test = numpy.array(test)
print test[0]
print test.shape
dfintest = xgboost.DMatrix(test,missing=float("nan"))
finpred = model.predict(dfintest)
print finpred
to_output.to_output(finpred,"predictions.csv")
