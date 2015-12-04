import csv
import gc
import numpy
import numpy.random as rand
import xgboost
import to_output
import sklearn.grid_search as grid
import sklearn.pipeline as pipe
import sklearn.feature_selection as select
from sklearn import svm

"""
Train with 80% of training set, test with 20% of training set to get
a measure of accuracy
"""
def test():
    #read the csv file into a numpy ndarray
    data = read("../nomissing.csv")
    print "Data read"
    rand.shuffle(data)
    split = numpy.floor(.8*data.shape[0])
    Xtrain = data[:split,:-1]
    ytrain = data[:split,-1]
    Xtest = data[split:,:-1]
    ytest = data[split:,-1]
    #put model here
    mod = model(Xtrain,ytrain)
    print numpy.mean(numpy.abs(mod.predict(Xtest) - ytest)/ytest)
    print mod.get_params()

"""
Train with entire training set, make predictions on test set to get
an acutal output for the competition
"""
def output():
    #read in training set
    train = numpy.genfromtxt("../nomissing.csv",delimiter=',')
    Xtrain = train[:,2:-1]
    ytrain = train[:,-1]
    print "Data read"
    #put model here
    mod = model(Xtrain,ytrain)
    print mod.get_params()
    #now test
    test = read("../avtest.csv")
    to_output.to_output(mod.predict(test),"predictions3.csv")


"""
Actual model to run
"""
def model(X,y):
    #feature selection
    classifier = svm.LinearSVR()
    selector = select.RFE(classifier, step=.2,n_features_to_select=5)
    Xt = selector.fit_transform(X,y)
    print "Data transformed"
    #create the model
    xg = xgboost.XGBRegressor()
    params = [{'max_depth':numpy.linspace(2,4,3).astype(int),
        'learning_rate':numpy.logspace(-2,0,3),
        'n_estimators':numpy.linspace(95,105,5).astype(int)}]
    gridres = grid.GridSearchCV(estimator=xg,param_grid=params,cv=3)
    return gridres.fit(Xt,y)


"""
Convenience method to read csv files. Takes a file name, returns a
numpy array with all the data
"""
def read(filename):
    inputreader = csv.reader(open(filename,'r'), delimiter=",")
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
    return data
