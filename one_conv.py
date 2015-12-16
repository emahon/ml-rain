import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.optimizers import SGD
from PIL import Image 
import theano
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils, generic_utils
import time
# One dimensional Convoutional Neural Net

data=np.genfromtxt('1D_Convolutional_Data.csv',delimiter=',')
X=np.reshape(data,[15219,60,5])
Y=np.genfromtxt('1D_guage_values.csv', delimiter=',')

# Break Test Train
vec=np.array(range(X.shape[0]))
vec.astype(np.int64)
tr=np.random.choice(vec, size=10000, replace=False)
tst=np.setdiff1d(vec,tr)
tst.astype(np.int64)
A=tr.tolist()
B=tst.tolist()
X_train=X[A,:,:]
X_test=X[B,:,:]
y_train=Y[A]
y_test=Y[B]
model=Sequential()
model.add(Convolution1D(5,6, border_mode='valid',input_shape=(60,5)))
model.add(Activation('tanh'))
model.add(MaxPooling1D(pool_length=2))

model.add(Flatten())
model.add(Dense(16))
model.add(Activation('tanh'))
model.add(Dense(1))
model.add(Activation('linear'))
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer='sgd')
model.fit(X_train,y_train, batch_size=100, nb_epoch=20,show_accuracy=True)
train_vals=model.predict(X_train,batch_size=100)
test_vals=model.predict(X_test,batch_size=100)
np.savetxt('1D_Test_Predictions_half.csv',test_vals)
np.savetxt('1D_Y_test_half.csv',y_test)
np.savetxt('1D_Train_Predictions_half.csv',train_vals)
np.savetxt('1D_Y_train_w_drop_half.csv',y_train)
