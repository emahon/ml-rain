"""
Run keras example for the heck of it. nomissing.csv comes from stats.py
"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np

if __name__ == '__main__':
	data = np.genfromtxt('nomissing.csv',delimiter=',')
	y = data[:,-1]
	X = data[:,2:-1]

	model = Sequential()
	# Dense(64) is a fully-connected layer with 64 hidden units.
	# in the first layer, you must specify the expected input data shape:
	# here, 20-dimensional vectors.
	model.add(Dense(64, input_dim=X.shape[1], init='uniform'))
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(64, init='uniform'))
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(2, init='uniform'))
	model.add(Activation('linear'))

	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='mean_squared_error', optimizer=sgd)

	model.fit(X_train, y_train, nb_epoch=20, batch_size=16)
	score = model.evaluate(X_test, y_test, batch_size=16)