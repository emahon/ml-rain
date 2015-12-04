"""
Performs Random Forest Regression on training and test data

Author: Ryan Gooch, 11/30/2015
"""

import numpy as np 
from scipy.sparse import csr_matrix
from sklearn import cross_validation,svm,metrics
from tempfile import mkdtemp
import os.path as path
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score

def getdata(fil) :
	data = np.genfromtxt('nomissing.csv',delimiter=',')
	y = data[:,-1]
	X = data[:,2:-1] # Peel off ID and minutes past, as well as y
	return X,y,data

def gettestdata(fil) :
	data = np.genfromtxt(fil,delimiter=',')
	return data

def writetest(Xpreds, fil='testresultsNN.csv') :
	import csv
	csv.field_size_limit(1000000000)
	outwriter = csv.writer(open(fil,'w'),delimiter=",")
	rows = np.arange(0,len(Xpreds))
	for row in rows :
		outwriter.writerow([row+1,Xpreds[row]])

def scalenans(X) :
	"""
	Scales 1-D feature vector, ignoring NaNs
	"""
	Xscale = (X - np.nanmean(X)) / np.nanstd(X)
	return Xscale

if __name__ == '__main__':
	X,y,data = getdata(fil = 'nomissing.csv')

	# print yorig.shape
	# filename = path.join(mkdtemp(), 'X.dat')
	# file2 = path.join(mkdtemp(), 'y.dat')
	# X = np.memmap(filename,mode='w+',shape=Xorig.shape,dtype='float32')
	# X[:] = Xorig[:]
	
	# y = np.memmap(file2,mode='w+',shape=yorig.shape,dtype='float32')
	# y[:] = yorig[:]
	# Take small sample
	# choices = np.random.choice(np.arange(0,X.shape[0]),size=X.shape[0],replace=False)
	rs = 2727

	# Need to scale data or else it blows up
	X = preprocessing.scale(X).copy()

	# Train/test split for comparison
	X_train, X_test, y_train, y_test = \
		cross_validation.train_test_split(X, y, \
			test_size=0.4, random_state=rs)

	# Big RFR
	reg = RandomForestRegressor(n_estimators=200, oob_score=True,
								random_state=rs,n_jobs=-1)
	reg.fit(X_train,y_train)
	preds = reg.predict(X_test)
	
	err = np.sum(np.abs(preds-y_test))/preds.shape[0]
	print err

	# Test it!
	Xtest = gettestdata('avtest.csv')
	Xtest = scalenans(Xtest).copy()
	


	# numtotest = X.shape[0]*2/3
	# classifier = svm.LinearSVR(C=0.1,dual=False,loss='squared_epsilon_insensitive',\
	# 	).fit(X[choices[:numtotest]], y[choices[:numtotest]])

	# n_estimators_list = np.array([5,10,25,50,100])
	# gammas = np.logspace(-5,-1,5)
	# classifier = GridSearchCV(estimator=RandomForestRegressor(), scoring='mean_absolute_error',\
	# 	param_grid=dict(n_estimators=n_estimators_list,n_jobs=[-1],\
	# 		)).fit(X[choices[:numtotest]], y[choices[:numtotest]])


	# print classifier.best_score_
	# print classifier.best_estimator_

	# preds = classifier.predict(X[choices[numtotest:]])
	# err = np.sum(np.abs(preds-y[choices[numtotest:]]))/preds.shape[0]
	# print err

	# classifier = RandomForestRegressor(random_state=rs, n_estimators=100).fit(\
	# 	X[choices[:numtotest]],y[choices[:numtotest]])
	# # score = cross_val_score(estimator, X, y, scoring='mean_absolute_error')
	# preds = classifier.predict(X[choices[numtotest:]])
	# err = np.sum(np.abs(preds-y[choices[numtotest:]]))/preds.shape[0]
	# print err

	# Xtestorig = gettestdata(fil = 'avtest.csv')
	# filename = path.join(mkdtemp(), 'X.dat')
	# Xtest = np.memmap(filename,mode='w+',shape=Xtestorig.shape,dtype='float32')
	# Xtest[:] = Xtestorig[:]
	# Xpreds = classifier.predict(Xtest)