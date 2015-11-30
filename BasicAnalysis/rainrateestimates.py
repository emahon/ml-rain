import numpy as np
from rainratefunctions import *

if __name__ == '__main__':
	fil = open('lessaverage.csv')
	data = np.genfromtxt(fil,delimiter=',')

	# Rain rate R(KDP)
	rr = np.apply_along_axis(R_KDP,1,data)

	# Filter out NaNs
	row = 0
	diffs = []
	while row < rr.shape[0] :
		if (np.isnan(rr[row,0]) == False) :
			diffs.append(np.abs(rr[row,1] - rr[row,0]))
		row = row + 1

	mse_rr = np.sum(np.array(diffs)**2) / rr.shape[0]
	print 'R(KDP) MSE = %s'%(mse_rr)

	# Rain rate R(ZDR, Zh)
	rr = np.apply_along_axis(R_ZDR_Zh,1,data)

	# Filter out NaNs
	row = 0
	diffs = []
	while row < rr.shape[0] :
		if (np.isnan(rr[row,0]) == False) :
			diffs.append(np.abs(rr[row,1] - rr[row,0]))
		row = row + 1

	mse_rr = np.sum(np.array(diffs)**2) / rr.shape[0]
	print 'R(ZDR,Zh) MSE = %s'%(mse_rr)

	# Rain rate R(Zh) nexrad
	rr = np.apply_along_axis(R_Zh_nexrad,1,data)

	# Filter out NaNs
	row = 0
	diffs = []
	while row < rr.shape[0] :
		if (np.isnan(rr[row,0]) == False) :
			diffs.append(np.abs(rr[row,1] - rr[row,0]))
		row = row + 1

	mse_rr = np.sum(np.array(diffs)**2) / rr.shape[0]
	print 'R(Zh) NEXRAD MSE = %s'%(mse_rr)

	# Rain rate R(Zh) marshall palmer
	rr = np.apply_along_axis(R_Zh_marshall_palmer,1,data)

	# Filter out NaNs
	row = 0
	diffs = []
	while row < rr.shape[0] :
		if (np.isnan(rr[row,0]) == False) :
			diffs.append(np.abs(rr[row,1] - rr[row,0]))
		row = row + 1

	mse_rr = np.sum(np.array(diffs)**2) / rr.shape[0]
	print 'R(Zh) Marshall Palmer MSE = %s'%(mse_rr)

	# Rain rate R(ZDR, KDP)
	rr = np.apply_along_axis(R_ZDR_KDP,1,data)

	# Filter out NaNs
	row = 0
	diffs = []
	while row < rr.shape[0] :
		if (np.isnan(rr[row,0]) == False) :
			diffs.append(np.abs(rr[row,1] - rr[row,0]))
		row = row + 1

	mse_rr = np.sum(np.array(diffs)**2) / rr.shape[0]
	print 'R(ZDR,KDP) MSE = %s'%(mse_rr)