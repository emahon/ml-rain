"""
Run keras example for the heck of it. nomissing.csv comes from stats.py
"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn import cross_validation, preprocessing
import numpy as np
import theano

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

	# Train/test split for comparison
	X_train, X_test, y_train, y_test = \
		cross_validation.train_test_split(X, y, \
			test_size=0.4, random_state=rs)


	model = Sequential()
	# Dense(64) is a fully-connected layer with 64 hidden units.
	# in the first layer, you must specify the expected input data shape:
	# here, 20-dimensional vectors.
	model.add(Dense(64, input_dim=X.shape[1], init='uniform'))
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(64, init='uniform',input_dim=64))
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(1, init='uniform',input_dim=64))
	model.add(Activation('linear'))

	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	
	# Use mean absolute error as loss function since that is 
	# what kaggle uses
	model.compile(loss='mean_absolute_error', optimizer=sgd)

	# Batch size = 100 seems to have stabilized it
	model.fit(X_train, y_train, nb_epoch=100, batch_size=10000)
	score = model.evaluate(X_test, y_test, batch_size=16)

	# Batch size | n_epoch | error_in | error_out | time 
	#	100			10  		4.25		4.29	01 m 00 s
	#	100			20			4.25		4.26	01 m 44 s						
	#	1000		10 			3.98		4.06	00 m 52 s
	#	10000 		10 			3.99 		4.04 	00 m 50 s
	#	10000		100 		3.98 		3.99 	06 m 11 s
