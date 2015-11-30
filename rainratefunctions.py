"""
Functions for various Rain Rate estimators in the literature.
There are many more than these, but these may be useful 
enough.

You'll need to have the processed data ready, as these are
made to only handle one sample at a time. You can use an
np.apply_along_axis though to apply them to the data all 
at once if you like.

Author: Ryan Gooch, 11/30/15
"""
import numpy as np

def R_ZDR_KDP (d) :
	"""
	R(ZDR, KDP)

	Rainfall rate (RR) estimation based on KDP and ZDR. From
	Bringi and Chandrasekar (2001), Section 8.1.3. 

	Input: Sample (row) from Kaggle Data
	Output: RR Estimate, RR Actual
	"""
	kdp = d[19]
	zdr = d[15]

	# Coeffecients
	c = 90.8
	a = 0.93
	b = -1.69

	# Return divided by 12 to get mm / (scan period)
	# instead of mm / h
	est = c * (np.abs(kdp) ** a) * 10 **(0.1 * b * zdr) / 12
	return est,d[-1]

def R_Zh_marshall_palmer (d) :
	"""
	R(Zh)

	Marshall Palmer RR estimator. 

	Input: Sample (row) from Kaggle Data
	Output: RR Estimate, RR Actual
	"""

	zh = d[7] # RefComposite

	# Coefficients
	c = 0.0365
	a = 0.625

	est = c * zh ** a / 12
	return est, d[-1]

def R_Zh_nexrad (d) :
	"""
	R(Zh)

	RR estimator used by WSR-88D radars

	Input: Sample (row) from Kaggle Data
	Output: RR Estimate, RR Actual
	"""

	zh = d[7] # RefComposite

	# Coefficients
	c = 0.017
	a = 0.714

	est = c * zh ** a / 12
	return est, d[-1]

def R_ZDR_Zh (d) :
	"""
	R(ZDR, Zh)

	Rainfall rate (RR) estimation based on Zh and ZDR. From
	Bringi and Chandrasekar (2001), Section 8.1.1. 
	Original Paper: Gorgucci et al, 1994

	Input: Sample (row) from Kaggle Data
	Output: RR Estimate, RR Actual
	"""
	zh = d[7]
	zdr = d[15]

	# Coeffecients
	c = 0.0067
	a = 0.93
	b = -3.43

	# Return divided by 12 to get mm / (scan period)
	# instead of mm / h
	est = c * (zh ** a) * 10 **(0.1 * b * zdr) / 12
	return est,d[-1]

def R_KDP (d) :
	def R_Zh_nexrad (d) :
	"""
	R(KDP)

	RR estimator using only KDP

	Input: Sample (row) from Kaggle Data
	Output: RR Estimate, RR Actual
	"""

	kdp = d[19] # RefComposite

	# Coefficients
	c = 50.7
	b = 0.85

	est = c * zh ** b / 12
	return est, d[-1]