"""
Trying to formulate a test set analysis system
"""

import numpy as np 
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn import cross_validation,svm,metrics
from tempfile import mkdtemp
import os.path as path
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import Imputer, scale
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import cross_val_score

def getdata(fil) :
	data = np.genfromtxt(fil,delimiter=',')
	y = data[:,-1]
	#spr.eliminate_zeros()
	return np.array(data),np.array(y),data

def gettestdata(fil) :
	data = np.genfromtxt(fil,delimiter=',')
	imp = Imputer(missing_values='NaN', strategy='median', axis=0)
	X = imp.fit_transform(data[:,2:])
	X = scale(X).copy()
	#spr.eliminate_zeros()
	return np.array(X)

def writetest(Xpreds, fil='testanswers.csv') :
	import csv
	csv.field_size_limit(1000000000)
	outwriter = csv.writer(open(fil,'w'),delimiter=",")
	rows = np.arange(0,len(Xpreds))
	for row in rows :
		outwriter.writerow([row+1,Xpreds[row]])

def refs() :
	return 3

if __name__ == '__main__':
	Xorig,yorig,data = getdata(fil = 'lessaverage.csv')

	Xdf = pd.DataFrame(Xorig[:,np.ix_([3])][:,0])
	Xdf = (Xdf - Xdf.mean()) / Xdf.std()

	print yorig.shape
	filename = path.join(mkdtemp(), 'X.dat')
	file2 = path.join(mkdtemp(), 'y.dat')
	X = np.memmap(filename,mode='w+',shape=Xdf.shape,dtype='float32')
	X[:] = Xdf[:]
	
	y = np.memmap(file2,mode='w+',shape=yorig.shape,dtype='float32')
	y[:] = yorig[:]
	# Take small sample
	choices = np.random.choice(np.arange(0,X.shape[0]),size=X.shape[0],replace=False)
	rs = 2727

	numtotest = 100000
	# classifier = svm.LinearSVR(C=0.1,dual=False,loss='squared_epsilon_insensitive',\
	# 	).fit(X[choices[:numtotest]], y[choices[:numtotest]])

	Cs = np.logspace(-1, 5, 7)
	gammas = np.logspace(-5,-1,5)
	classifier = GridSearchCV(estimator=svm.LinearSVR(), scoring='mean_absolute_error',\
		param_grid=dict(C=Cs,epsilon=[0],dual=[False],loss=['squared_epsilon_insensitive']\
			)).fit(X[choices[:numtotest]], y[choices[:numtotest]])

	print classifier.best_score_
	print classifier.best_estimator_

	preds = classifier.predict(X[choices[numtotest:]])
	err = np.sum(np.abs(preds-y[choices[numtotest:]]))/preds.shape[0]
	print err