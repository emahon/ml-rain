rom keras.optimizers import SGD
from PIL import Image 
import theano
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils, generic_utils
import time




# Load Images Put them into Test and Train Set
# Compile indvidual images into a 4D Theano Tensor
t1=time.clock()
im1=Image.open('Rain_B_1.png')
im2=Image.open('Rain_P_1.png')
im3=Image.open('Rain_M_1.png')
#im4=Image.open('Rain_P_1.png')
#im5=Image.open('Rain_M_1.png')
x1=np.array(im1)
m_val=float(x1.max())
x1=x1/m_val
x1=x1[None,:,:]
x2=np.array(im2)
m_val=float(x2.max())
x2=x2/m_val
x2=x2[None,:,:]
X=np.append(x1,x2,axis=0)
x3=np.array(im3)
m_val=float(x3.max())
x3=x3/m_val
x3=x3[None,:,:]
X=np.append(X,x3,axis=0)
#x4=np.array(im4)
#m_val=float(x4.max())
#x4=x4/m_val
#x4=x4[None,:,:]
#X=np.append(X,x4,axis=0)
#x5=np.array(im5)
#m_val=float(x5.max())
#x5=x5/m_val
#x5=x5[None,:,:]
#X=np.append(X,x5,axis=0)
X=X[None,:,:,:]
N=15219
for i in range(2,N):
	#f1='Rain_R_'+str(i)+'.png'
	f1='Rain_B_'+str(i)+'.png'
	#f3='Rain_G_'+str(i)+'.png'
	f2='Rain_P_'+str(i)+'.png'
	f3='Rain_M_'+str(i)+'.png'
	im1=Image.open(f1)
	im2=Image.open(f2)
	im3=Image.open(f3)
	#im4=Image.open(f4)
	#im5=Image.open(f5)
	x1=np.array(im1)
	m_val=float(x1.max())
	x1=x1/m_val
	x1=x1[None,:,:]
	x2=np.array(im2)
	m_val=float(x2.max())
	x2=x2/m_val
	x2=x2[None,:,:]
	Xi=np.append(x1,x2,axis=0)
	x3=np.array(im3)
	m_val=float(x3.max())
	x3=x3/m_val
	x3=x3[None,:,:]
	Xi=np.append(Xi,x3,axis=0)
	#x4=np.array(im4)
	#m_val=float(x4.max())
	#x4=x4/m_val
	#x4=x4[None,:,:]
	#Xi=np.append(Xi,x4,axis=0)
	#x5=np.array(im5)
	#m_val=float(x5.max())
	#x5=x5/m_val
	#x5=x5[None,:,:]
	#Xi=np.append(Xi,x5,axis=0)
	Xi=Xi[None,:,:,:]
	X=np.append(X,Xi,axis=0)
	


Y=np.genfromtxt('5_image_GUAGE_VALS.csv',delimiter=',')
# Break into test and train start with 900 images for train
vec=np.array(range(X.shape[0]))
vec.astype(np.int64)
tr=np.random.choice(vec,size=10000,replace=False)
tr.astype(np.int64)
tst=np.setdiff1d(vec,tr)
print(type(tst))
tst.astype(np.int64)
A=tr.tolist()
B=tst.tolist()
print(type(A))
X_train=X[A,:,:,:]
y_train=Y[A]
X_test=X[B,:,:,:]
y_test=Y[B]
#y_train, y_test = [np_utils.to_categorical(x) for x in (y_train, y_test)]
print('Data Load, test, train complete')
model=Sequential()
#Convolutional Layer
model.add(Convolution2D(10,3,3, border_mode='valid', input_shape=(3,23,23)))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(16))
model.add(Activation('tanh'))
model.add(Dropout(0.25)) # add a dropout to combat overfitting

model.add(Dense(1))
model.add(Activation('linear'))
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer='sgd')
model.fit(X_train,y_train, batch_size=500, nb_epoch=10,show_accuracy=True)
train_vals=model.predict(X_train,batch_size=500)
test_vals=model.predict(X_test, batch_size=500)
#print(type(train_vals))
#print(test_vals)
np.savetxt('RAIN_Train_BPM.csv',np.array(train_vals))
np.savetxt('RAIN_Test_BPM.csv',np.array(test_vals))
np.savetxt('Y_test_BPM.csv',y_test)
np.savetxt('Y_train_BPM.csv',y_train)
#score=model.evaluate(X_test,y_test,batch_size=100,show_accuracy=True)	
#print(score)
t2=time.clock()
print((t2-t1))
