import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from PIL import Image 
import theano
from sklearn.cross_validation import train_test_split
import MADCALC as MAD



# Load Images Put them into Test and Train Set
im=Image.open('Rain_Images_reflectivity_1.png')
im=np.array(im)
m_val=float(im.max())
im=im/m_val
im=im[None,:,:]
X=im
N=1001
for i in range(1,N):
	file_name='Rain_Images_reflectivity_'+str(i)+'.png'
	im=Image.open(file_name)
	im=np.array(im)
	im=im[None,:,:]
	m_val=float(im.max())
	im=im/m_val
	X=np.append(X,im,axis=0)

print(X.shape)

Y=np.genfromtxt('rain_measures.csv',delimiter=',')
# Break into test and train start with 900 images for train
vec=np.linspace(0,1000,1001)
vec.astype(np.int64)
tr=np.random.choice(vec,size=900,replace=False)
tr.astype(np.int64)
tst=np.setdiff1d(vec,tr)
print(type(tst))
tst.astype(np.int64)
A=tr.tolist()
B=tst.tolist()
print(type(A))
X_train=X[A,:,:]
X_train=X_train[:,None,:,:]
y_train=Y[A]
X_test=X[B,:,:]
X_test=X_test[:,None,:,:]
y_test=Y[B]
model=Sequential()
#Convolutional Layer
model.add(Convolution2D(3,5,5, border_mode='valid', input_shape=(1,23,23)))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(2))
model.add(Activation('tanh'))
model.add(Dense(1))
model.add(Activation('linear'))
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
model.fit(X_train,y_train, batch_size=100, nb_epoch=5)
train_vals=model.predict(X_train, batch_size=100)
test_vals=model.predict(X_test, batch_size=100)
madval1=MAD.MADCALC(train_vals,y_train)
madval2=MAD.MADCALC(test_vals,y_test)
print madval1,madval2
print(test_vals)
	
