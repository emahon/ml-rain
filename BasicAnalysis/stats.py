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

def plot_corr_matrix(X) :
	"""
	Plots correlation matrix for data
	"""
	import numpy as np
	import pandas as pd
	import seaborn as sns
	import matplotlib.pyplot as plt

	sns.set(style="white")

	labels = ["radar dist",
				"Ref",
				"Ref 5x5 10th",
				"Ref 5x5 50th",
				"Ref 5x5 90th",
				"RefComposite",
				"RefComposite 5x5 10th",
				"RefComposite 5x5 50th",
				"RefComposite 5x5 90th",
				"Rho_HV",
				"Rho_HV 5x5 10th",
				"Rho_HV 5x5 50th",
				"Rho_HV 5x5 90th",
				"Zdr",
				"Zdr 5x5 10th",
				"Zdr 5x5 50th",
				"Zdr 5x5 90th",
				"Kdp",
				"Kdp 5x5 10th",
				"Kdp 5x5 50th",
				"Kdp 5x5 90th",
				"Expected"]

	d = pd.DataFrame(data=X[:,2:].copy(), columns=labels)

	# Compute the correlation matrix
	corr = d.corr()

	# Set up the matplotlib figure
	f, ax = plt.subplots(figsize=(11, 9))

	sns.corrplot(d)
	ax.set_title('Correlation Matrix for Radar Features and Output Variable')
	ax.set_xlabel('Features (along diagonal)')
	ax.set_ylabel('Correlation Values (upper triangle)')
	f.tight_layout()
	plt.show()

	# sns.pairplot(d, hue="Quality")
	# f.tight_layout()
	# plt.show()

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
	ynonans = []
	for row in np.arange(0,X.shape[0]) :
		if nonemissing(X[row]) == False :
			Xnonans.append(X[row])
			ynonans.append([y[row]])

	# Array format
	Xn = np.array(Xnonans) # Still has expected values in final column
	yn = np.array(ynonans)
	# Pure training set 
	# writetest(Xn,'nomissing.csv')

	# Plot Correlation Matrix
	# plot_corr_matrix(Xn)

	