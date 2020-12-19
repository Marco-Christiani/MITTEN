import numpy as np
from numpy import matmul, sqrt, array

def mcusum(df,
		   num_in_control,
		   k,):
	"""
	Implementation of the Multivariate Cumulative Sum (MCUSUM) method.

		Based on Kent(2007) https://etd.ohiolink.edu/rws_etd/send_file/send?accession=kent1185558637&disposition=inline

		- Reference 17 : (Jackson 1985)
		
		- Reference 5  : (Crosier 1988)

	Args:
		df: multivariate dataset as Pandas DataFrame
		num_in_control: number of in control observations
		k: the slack parameter which determines model sensetivity (should typically be set to 1/2 of the mean shift that you expect to detect)
	Returns:
		MCUSUM statistic values as a list
	"""
	a = df.head(num_in_control).mean(axis=0)  # mean vector of in control data
	cov_inv = np.linalg.inv(np.cov(df.head(num_in_control), rowvar=False,bias=True))  # covariance matrix inverted
	s_old = [0] * df.shape[1]
	y_vals = [0] * df.shape[0]
	for n in range(0, df.shape[0]):
		# get current line
		x = df.iloc[n]

		# calculate Cn
		tf = array(s_old + x - a)
		c = matmul(matmul(tf, cov_inv), tf)
		c = sqrt(c)

		# calculate kv (vector k)
		kv = (k / c) * tf

		# calculate new s
		if (c <= k):
			s_old = 0
		else:
			s_old = tf * (1 - (k / c))
			# calculate y (new c)
			tf = array(s_old + x - a)
			y = matmul(matmul(tf, cov_inv), tf)
			y = sqrt(y)
			y_vals[n] = y

	#returns the mcusum statistics as a list
	return y_vals
