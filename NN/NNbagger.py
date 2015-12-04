from sklearn.ensemble import BaggingRegressor
from sklearn import cross_validation, preprocessing, metrics

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.regularizers import l2

import numpy as np
import theano
import matplotlib.pyplot as plt

class nnbagger:

	"""
	Sets up Neural Network as an object BaggingRegressor
	understands...?
	"""
	def __init__(self) :
		# Set up the NN here
		self.nnsetup()

	def nnsetup(self):
		"""
		Sets up the Network
		"""
		model = Sequential()
		# Dense(64) is a fully-connected layer with 64 hidden units.
		# in the first layer, you must specify the expected input data shape
		model.add(Dense(32, input_dim=X.shape[1], init='he_normal'))#, W_regularizer=l2(0.1)))
		model.add(Activation('tanh'))
		model.add(Dropout(0.5))
		model.add(Dense(16, init='he_normal',input_dim=32))#, W_regularizer=l2(0.1)))
		model.add(Activation('tanh'))
		model.add(Dropout(0.5))
		model.add(Dense(1, init='he_normal',input_dim=16))#, W_regularizer=l2(0.1)))
		model.add(Activation('linear'))

		sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
		
		# Use mean absolute error as loss function since that is 
		# what kaggle uses
		model.compile(loss='mean_absolute_error', optimizer=sgd)

	def fit(self,X,y) :
		"""
		Fit method for classifier
		"""
		model.fit(X, y, nb_epoch=10, batch_size=1000)

	def predict(self,X,y) :
		"""
		Performs predictions given data
		"""
		preds = model.predict(Xtest, batch_size=16, verbose=1)
		return preds

def scalenans(X) :
	"""
	Scales 1-D feature vector, ignoring NaNs
	"""
	Xscale = (X - np.nanmean(X)) / np.nanstd(X)
	return Xscale

def getdata(fil) :
	data = np.genfromtxt(fil,delimiter=',')
	return data

def writetest(Xpreds, fil='testresultsNN.csv') :
	import csv
	csv.field_size_limit(1000000000)
	outwriter = csv.writer(open(fil,'w'),delimiter=",")
	rows = np.arange(0,len(Xpreds))
	for row in rows :
		outwriter.writerow([row+1,Xpreds[row]])

if __name__ == '__main__':
	data = getdata("nomissing.csv")
	y = data[:,-1]
	X = data[:,2:-1] # Peel off ID and minutes past, as well as y

	#Set random state\
	rs = 19683

	# enable on-the-fly graph computations
	theano.config.compute_test_value = 'warn'

	# Need to scale data or else it blows up
	X = preprocessing.scale(X).copy()

	# Build regressor 
	model = BaggingRegressor(base_estimator=nnbagger(),
						n_estimators=10,
						bootstrap=True,
						random_state=rs,
						verbose=1)