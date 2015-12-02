import numpy as np
import os.path as path

from tempfile import mkdtemp
from sklearn.preprocessing import scale
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation,svm,metrics

def getdata(fil) :
	data = np.genfromtxt(fil,delimiter=',')
	y = data[:,-1]
	return data,y

def gettestdata(fil) :
	data = np.genfromtxt(fil,delimiter=',')
	return data

def classify_GS(X,y,rs=19683) :
	Xorig = X
	yorig = y

	Xorig = scale(Xorig).copy()

	# Use memory map to prevent ram breaking
	file1 = path.join(mkdtemp(), 'X.dat')
	file2 = path.join(mkdtemp(), 'y.dat')
	X = np.memmap(file1,mode='w+',shape=Xorig.shape,dtype='float32')
	X[:] = Xorig[:]

	y = np.memmap(file2,mode='w+',shape=yorig.shape,dtype='float32')
	y[:] = yorig[:]

	# Shuffle data
	np.random.seed(rs)
	choices = np.random.choice(np.arange(0,X.shape[0]),size=X.shape[0],replace=False)
	
	# Run a gridsearch on a sample of the data
	numtotest = X.shape[0] / 2
	Cs = np.logspace(-1, 5, 7)
	classifier = GridSearchCV(estimator=svm.LinearSVR(), scoring='mean_absolute_error',\
		param_grid=dict(C=Cs,dual=[False],\
			loss=['squared_epsilon_insensitive']\
			)).fit(X[choices[:numtotest]], y[choices[:numtotest]])

	print classifier.best_score_
	print classifier.best_estimator_

	preds = classifier.predict(X[choices[numtotest:]])
	err = np.sum(np.abs(preds-y[choices[numtotest:]]))/preds.shape[0]
	print err

	return classifier

def dotest(classifier, X):
	return classifier.predict(X)

def writetest(Xpreds, fil='testanswersSVR.csv') :
	import csv
	csv.field_size_limit(1000000000)
	outwriter = csv.writer(open(fil,'w'),delimiter=",")
	rows = np.arange(0,len(Xpreds))
	for row in rows :
		outwriter.writerow([row+1,Xpreds[row]])

if __name__ == '__main__':
	
	# dumb names for priority matrices. update later if desired
	a = []
	ay = []
	b = []
	by = []
	c = []
	cy = []
	d = []
	dy = []
	e = []
	ey = []

	# Ran master processing, filtered out Ref > 250 but no split
	X, y = getdata('average.csv')
	X = X[:,np.ix_([2,3,15,19])].copy()[:,0] # radardist, Ref, ZDR, KDP
	for row in np.arange(0,len(X)) :
		# Top priority. Ref, ZDR, KDP
		if np.isnan(X[row]).any() == False : # none are missing
			a.append(X[row])
			ay.append(y[row])

		# Second priority. Ref, KDP
		elif np.isnan(X[row,np.ix_([0,1,3])]).any() == False : 
			b.append(X[row,np.ix_([0,1,3])][0])
			by.append(y[row])

		# Third priority. ZDR, KDP
		elif np.isnan(X[row,np.ix_([0,2,3])]).any() == False : 
			c.append(X[row,np.ix_([0,2,3])][0])
			cy.append(y[row])

		# Fourth priority. Ref, ZDR
		elif np.isnan(X[row,np.ix_([0,1,2])]).any() == False : 
			d.append(X[row,np.ix_([0,1,2])][0])
			dy.append(y[row])

		# Fifth priority. Ref
		elif np.isnan(X[row,1]).any() == False : 
			e.append(X[row,np.ix_([0,1])][0].copy())
			ey.append(y[row])

		# Else, it is garbage and not graded
		# But as this is training we need not worry about it

	# Make every thing arrays
	a = np.array(a).copy()
	b = np.array(b).copy()
	c = np.array(c).copy()
	d = np.array(d).copy()
	e = np.array(e).copy()

	ay = np.array(ay).copy()
	by = np.array(by).copy()
	cy = np.array(cy).copy()
	dy = np.array(dy).copy()
	ey = np.array(ey).copy()

	# Priority 2 never shows up actually. Breakdown for the others:
	# Priority:		|	Number:
	#		1		|	511520
	#		3		|	23087
	#		4		|	40698
	#		5 		|	170850

	classifier_a = classify_GS(a,ay)
	classifier_c = classify_GS(c,cy)
	classifier_d = classify_GS(d,dy)
	classifier_e = classify_GS(e,ey)

	# Now generate a test
	X= gettestdata('avtest.csv')
	X = X[:,np.ix_([2,3,15,19])][:,0] # radardist, Ref, ZDR, KDP
	
	Xorig = X
	Xorig = scale(Xorig).copy()
	file1 = path.join(mkdtemp(), 'Xorig.dat')
	X = np.memmap(file1,mode='w+',shape=Xorig.shape,dtype='float32')
	X[:] = Xorig[:]

	res = []
	for row in np.arange(0,len(X)) :
		# Top priority. Ref, ZDR, KDP
		if np.isnan(X[row]).any() == False : # none are missing
			temp = dotest(classifier_a, X[row])
			res.append(temp[0])

		# Third priority. ZDR, KDP
		elif np.isnan(X[row,np.ix_([0,2,3])][0]).any() == False : 
			temp = dotest(classifier_c, X[row,np.ix_([0,2,3])][0])
			res.append(temp[0])

		# Fourth priority. Ref, ZDR
		elif np.isnan(X[row,np.ix_([0,1,2])][0]).any() == False : 
			temp = dotest(classifier_d, X[row,np.ix_([0,1,2])][0])
			res.append(temp[0])

		# Fifth priority. Ref
		elif np.isnan(X[row,np.ix_([0,1])][0]).any() == False : 
			temp = dotest(classifier_e, X[row,np.ix_([0,1])][0])
			res.append(temp[0])

		# Else, it is garbage and not graded
		else :
			res.append(0)

	# # Make everything arrays. Probably unnecessary here though
	# res = np.array(res).copy()
	# print res.shape

	writetest(res,'SVRtest1.csv')