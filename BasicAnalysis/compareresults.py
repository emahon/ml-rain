import numpy as np 

def writetest(Xpreds, fil='testresultsNN.csv') :
	import csv
	csv.field_size_limit(1000000000)
	outwriter = csv.writer(open(fil,'w'),delimiter=",")
	rows = np.arange(0,len(Xpreds))
	for row in rows :
		outwriter.writerow([row+1,Xpreds[row]])

if __name__ == '__main__':
	reg1data = np.genfromtxt('RFR.200.60pct.standardized.csv',
							delimiter=',',skip_header=1)
	reg1 = reg1data[:,-1] # Just get preds

	reg2data = np.genfromtxt('testresultsNNallstandardized.csv',
							delimiter=',',skip_header=1)
	reg2 = reg2data[:,-1] # Just get preds

	reg3data = np.genfromtxt('NN.64.32.60pct.csv',
							delimiter=',',skip_header=1)
	reg3 = reg3data[:,-1] # Just get preds

	# 2 and 3 here closely correlated, 1 not so much

	reg4data = np.genfromtxt('NN.128.64.32.60pct.csv',
							delimiter=',',skip_header=1)
	reg4 = reg4data[:,-1] # Just get preds

	alltests = np.transpose(np.array([reg1,reg2,reg3,reg4]))
	print np.corrcoef(alltests)

	# Best so far was 64.32.60pct.csv. Give it weight of two,
	# all others weight 1
	ensembleaverage = []
	for row in np.arange(0,len(alltests)) :
		temp = reg3[row] * 2 + reg1[row] + reg2[row] + reg4[row]
		ensembleaverage.append(temp/5)

	ea = np.array(ensembleaverage)
	writetest(ea,'ensembleaverage.4tests.csv')