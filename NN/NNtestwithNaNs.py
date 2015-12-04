"""
Same training as kerasNN but will test data.
Imputes missing values with 0 after scaling.
Custom scaling had to be written to accommodate this.

Ryan Gooch
"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.regularizers import l2
from sklearn import cross_validation, preprocessing, metrics
import numpy as np
import theano
import matplotlib.pyplot as plt

def scalenans(X) :
	"""
	Scales 1-D feature vector, ignoring NaNs
	"""
	Xscale = (X - np.nanmean(X)) / np.nanstd(X)
	return Xscale

def vecnorm(X) :
	"""
	Scales data to -1 to 1 range
	"""
	Xtemp = X - np.min(X)
	Xnorm = Xtemp * 2 / np.max(Xtemp) - 1
	return Xnorm

def normnans(X) :
	"""
	Normalizes data if nans presents
	"""
	temp = X - np.nanmin(X)
	Xnorm = temp * 2 / np.nanmax(temp)
	return Xnorm

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

def genlabels(y, nbins = 100) :
	""" 
	Used for multilabel classification scheme.

	Takes array of actual outpus and assigns labels based
	on values. default number of bins = 100.

	Returns the bins and the labels
	"""
	bins = np.linspace(y.min(),y.max(),nbins)
	yd = np.digitize(y,bins,right=True)
	ynew = []
	for row in np.arange(0,len(yd)) :
		ynew.append(bins[yd[row]])

	return bins, ynew

if __name__ == '__main__':
	data = np.genfromtxt('nomissing.csv',delimiter=',')
	y = data[:,-1]
	X = data[:,2:-1] # Peel off ID and minutes past, as well as y

	#Set random state\
	rs = 19683

	# enable on-the-fly graph computations
	theano.config.compute_test_value = 'warn'

	# Need to scale data or else it blows up
	X = preprocessing.scale(X).copy()
	# X = vecnorm(X).copy()
	# print X.max()

	# Log Transform y
	# y = np.log(y).copy()

	# Bin the labels for multi-label classification
	# bins, y = genlabels(yorig, nbins = 100)

	# Train/test split for comparison
	X_train, X_test, y_train, y_test = \
		cross_validation.train_test_split(X, y, \
			test_size=0.4, random_state=rs)

	# Use ALL the training data this time
	# X_train = X
	# y_train = yorig

	model = Sequential()
	# Dense(64) is a fully-connected layer with 64 hidden units.
	# in the first layer, you must specify the expected input data shape
	model.add(Dense(64, input_dim=X.shape[1], init='he_normal'))#, W_regularizer=l2(0.1)))
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(32, init='he_normal',input_dim=64))#, W_regularizer=l2(0.1)))
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(1, init='he_normal',input_dim=32))#, W_regularizer=l2(0.1)))
	model.add(Activation('linear'))

	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	
	# Use mean absolute error as loss function since that is 
	# what kaggle uses
	model.compile(loss='mean_absolute_error', optimizer=sgd)

	# Batch size = 100 seems to have stabilized it
	model.fit(X_train, y_train, nb_epoch=10, batch_size=1000)
	# score = model.evaluate(X_test, y_test, batch_size=1000)
	# preds = model.predict(X_test, batch_size=1000, verbose=1)
	# print score
	# Now import averaged test data
	Xtest = gettestdata('avtest.csv')
	Xtest = Xtest[:,2:].copy()
	Xtest_temp = np.empty((Xtest.shape))
	# Scale, ignoring NaNs
	for col in np.arange(0,Xtest.shape[1]) :
		Xtest_temp[:,col] = scalenans(Xtest[:,col])

	# Now impute zeros in for NaNs
	Xtest = np.where(np.isnan(Xtest_temp),0,Xtest_temp)

	# OK, let's try to test this. Gonna be blind for the first
	# time due to limited time for the submission before 
	# we lose it.
	preds = model.predict(Xtest, batch_size=16, verbose=1)

	# Now save it!
	writetest(preds[:,0], fil='testresultsNN.60pct.standardized.csv')