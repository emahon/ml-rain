import numpy as np 

reg1data = np.genfromtxt('RFR.200.60pct.standardized.csv',delimiter=',',skiprows=1)
reg1 = reg1data[:,-1] # Just get preds

reg2data = np.genfromtxt('testresultsNNallstandardized.csv',delimiter=',',skiprows=1)
reg2 = reg2data[:,-1] # Just get preds

reg3data = np.genfromtxt('testresults.NN.60pct.standardized.csv',delimiter=',',skiprows=1)
reg3 = reg3data[:,-1] # Just get preds

# 2 and 3 here closely correlated, 1 not so much