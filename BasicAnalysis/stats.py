"""
Wanted to do some simple stats on the predictors

Ryan Gooch
"""

import numpy as np
import matplotlib.pyplot as plt 

def getdata(fil) :
	data = np.genfromtxt(fil,delimiter=',')
	y = data[:,-1]
	return data,y

def percentmissing(X) :
	"""
	Calculates percentage of missing values in 1-D array
	"""

	nans = np.count_nonzero(np.isnan(X))
	return float(nans) / len(X)

def nonemissing(X) :
	"""
	returns False if no nans present
	"""

	return np.isnan(X).any()

def writetest(Xpreds, fil='testanswers.csv') :
	import csv
	csv.field_size_limit(1000000000)
	outwriter = csv.writer(open(fil,'w'),delimiter=",")
	rows = np.arange(0,len(Xpreds))
	for row in rows :
		outwriter.writerow([row+1,Xpreds[row]])

if __name__ == '__main__':
	# Get data
	X, y = getdata('average.csv')

	kdp = X[:,19]
	zdr= X[:,15]
	zh = X[:,3]

	# Calculate percentage of feature missing in test data
	kdpnans = percentmissing(kdp) # 54.88%
	zdrnans = percentmissing(zdr) # 50.60%
	zhnans = percentmissing(zh) # 40.62%

	# Percentage missing from average.csv
	# kdp = 51.58%
	# zdr = 47.09%
	# zh = 34.52%

	# Compile rows that have no nans for pure dataset
	Xnonans = []
	for row in np.arange(0,X.shape[0]) :
		if nonemissing(X[row]) == False :
			Xnonans.append(X[row])

	# Array format
	Xn = np.array(Xnonans)

	# Pure training set 
	writetest(Xn,'nomissing.csv')