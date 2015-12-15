import xgboost
from sklearn import cross_validation
import numpy

data = numpy.genfromtxt("../nomissing.csv",delimiter=',')
X = data[:,2:-1]
y = data[:,-1]

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=19683)

dtrain = xgboost.DMatrix(X_train,label=y_train,missing=float("nan"))
params = dict()
model = xgboost.train(params,dtrain)
dtest = xgboost.DMatrix(X_test)
preds = model.predict(dtest)
score = numpy.abs(preds - y_test).sum()/y_test.shape[0]
print score
