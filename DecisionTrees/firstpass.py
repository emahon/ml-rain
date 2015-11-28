import csv
import numpy
import numpy.random as rand
import xgboost

#read the csv file into a numpy ndarray
inputfile = open("../moreaverage.csv",'r')
inputreader = csv.reader(inputfile, delimiter=",")
data = []
for row in inputreader:
    #convert strings to floats
    converted = [float(j) for j in row if (len(j) > 0)]
    data.append(converted)

data = numpy.array(data)
#split into test and train
rand.shuffle(data)
split = numpy.floor(.8*data.shape[0])
Xtrain = data[:split,:-1]
ytrain = data[:split,-1]
Xtest = data[split:,:-1]
ytest = data[split:,-1]
dtrain = xgboost.DMatrix(Xtrain,label=ytrain,missing=float("nan"))
params = dict()
model = xgboost.train(params,dtrain)
dtest = xgboost.DMatrix(Xtest, missing=float("nan"))
pred = model.predict(dtest)
vals = numpy.abs(pred - ytest)/ytest
print numpy.mean(vals)
