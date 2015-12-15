import svrcomp
import to_output
import numpy as np
from sklearn import cross_validation,svm,metrics
from sklearn.grid_search import GridSearchCV

if __name__=='__main__':
    Xtrain,ytrain,data = svrcomp.getdata("nomissing.csv")
    Xtest = svrcomp.gettestdata("../avtest.csv")
    Cs = np.logspace(-1, 5, 7)
    gammas = np.logspace(-5,-1,5)
    classifier = GridSearchCV(estimator=svm.LinearSVR(), scoring='mean_absolute_error',\
        param_grid=dict(C=Cs,epsilon=[0],dual=[False],loss=['squared_epsilon_insensitive']\
        ))
    classifier.fit(Xtrain,ytrain)
    preds = classifier.predict(Xtest)
    for i in range(len(preds)):
        if (preds[i] < 0):
            preds[i] = 0
    to_output.to_output(preds,"svrtest.csv")
    
    
