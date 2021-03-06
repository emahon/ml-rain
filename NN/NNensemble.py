from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.layers.convolutional import Convolution2D, MaxPooling2D
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

def NNrandstatecheck(X_train,y_train,X_test) :
	model = Sequential()
	# 100 neuron, 1 layer
	model.add(Dense(100, input_dim=X.shape[1], init='he_normal'))#, W_regularizer=l2(0.1)))
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(1, init='he_normal',input_dim=100))#, W_regularizer=l2(0.1)))
	model.add(Activation('linear'))

	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	
	# Use mean absolute error as loss function since that is 
	# what kaggle uses
	model.compile(loss='mean_absolute_error', optimizer=sgd)

	# Batch size = 100 seems to have stabilized it
	model.fit(X_train, y_train, nb_epoch=10, batch_size=1000)
	preds1 = model.predict(X_test, batch_size=1000, verbose=1)
	# print score

	model = Sequential()
	# 64 - 32 2-layer
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
	preds2 = model.predict(X_test, batch_size=1000, verbose=1)

	model = Sequential()
	# 128-64-32 2-layer
	model.add(Dense(128, input_dim=X.shape[1], init='he_normal'))#, W_regularizer=l2(0.1)))
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(64, init='he_normal',input_dim=128))#, W_regularizer=l2(0.1)))
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
	preds3 = model.predict(X_test, batch_size=1000, verbose=1)
	return preds1,preds2,preds3

def stacked(X_train,y_train,X_test) :
	"""
	Fat stacks of models!
	"""

	# NN to generate features
	model = Sequential()

	# input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
	# this applies 32 convolution filters of size 3x3 each.
	model.add(Dense(128, input_dim=X_train.shape[1], init='he_normal'))#, W_regularizer=l2(0.1)))
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(64, init='he_normal',input_dim=128))#, W_regularizer=l2(0.1)))
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(32, init='he_normal',input_dim=64))#, W_regularizer=l2(0.1)))
	finfeats = model.add(Activation('tanh'))
	model.add(Dense(1, init='he_normal',input_dim=32))#, W_regularizer=l2(0.1)))
	model.add(Activation('linear'))

	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='mean_absolute_error', optimizer=sgd)
	model.fit(X_train, y_train, nb_epoch=10, batch_size=1000,
					validation_split = 0.2, shuffle=True)
	# preds = model.predict(X_test, batch_size=1000, verbose=1)
	
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

	preds1,preds2,preds3 = NNrandstatecheck(X_train,y_train,X_test)
	print np.corrcoef(np.array([preds1[:,0],preds2[:,0],preds3[:,0]]))
