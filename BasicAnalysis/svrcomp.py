"""
Script computes support vector regression (SVR) for 
RR estimation using memory maps to handle large amount
of data. The missing values are set to 0 for 
simplicity before analysis. Ideally these should be
at least imputed though, which may be a useful feature
to add.

The script uses GridSearchCV to determine optimal SVR
parameters, then makes predictions based on this.
MSE is used as the scoring criterion.

Author: Ryan Gooch, 11/30/2015
"""

import numpy as np 
from scipy.sparse import csr_matrix
from sklearn import cross_validation,svm,metrics
from tempfile import mkdtemp
import os.path as path
from sklearn.grid_search import GridSearchCV

def getdata(fil) :
	data = np.genfromtxt(fil,delimiter=',',filling_values=0)
	data = np.where(np.isnan(data), 0, data).copy()
	spr = csr_matrix(data[2:,:-1])
	y = data[:,-1].copy()
	#spr.eliminate_zeros()
	return spr,y,data

if __name__ == '__main__':
	Xorig,yorig,data = getdata(fil = 'lessaverage.csv')

	print yorig.shape
	filename = path.join(mkdtemp(), 'X.dat')
	file2 = path.join(mkdtemp(), 'y.dat')
	X = np.memmap(filename,mode='w+',shape=Xorig.shape,dtype='float32')
	X[:] = Xorig.toarray()[:]
	
	y = np.memmap(file2,mode='w+',shape=yorig.shape,dtype='float32')
	y[:] = yorig[:]
	# Take small sample
	choices = np.random.choice(np.arange(0,X.shape[0]),size=X.shape[0],replace=False)
	rs = 2727

	# classifier = svm.SVR(kernel='rbf').fit(X[choices[:10000]], y[choices[:10000]])

	Cs = np.logspace(-2, 3, 6)
	gammas = np.logspace(-3,3,7)
	classifier = GridSearchCV(estimator=svm.SVR(), scoring='mean_squared_error',\
		param_grid=dict(C=Cs,gamma=gammas,kernel=['rbf'])).fit(\
			X[choices[:10000]], y[choices[:10000]])


	print classifier.best_score_
	print classifier.best_estimator_

	# preds = classifier.predict(X[choices[10000:]])
	# err = np.sum((preds-y[choices[10000:]])**2)/preds.shape[0]