import numpy as np
from numpy import matmul, sqrt, array
from .plotting import threshold_plot

def mcusum(df, num_in_control,k,alpha=.01, plotting=True, save="", plot_title="MCUSUM"):
	"""
	Implementation of the Multivariate Cumulative Sum (MCUSUM) method.

		Based on Kent(2007) https://etd.ohiolink.edu/rws_etd/send_file/send?accession=kent1185558637&disposition=inline

		- Reference 17 : (Jackson 1985)
		
		- Reference 5  : (Crosier 1988)

	Args:
		df: multivariate dataset as Pandas DataFrame
		num_in_control: number of in control observations
		k: the slack parameter which determines model sensetivity (should typically be set to 1/2 of the mean shift that you expect to detect)
		alpha: the percentage of false positives we want to allow, used for calculating the Upper Control Limit (default .01 or 1%)
		save: the directory to save the graphs to, if not changed from default, nothing will be saved
		plot_title: the title for the plot generated
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
	
	#calculate UCL
	in_stats = y_vals[0:num_in_control]
  	ucl = max(in_stats)
 
  	count = len([i for i in in_stats if i > ucl]) 

  	while(count < (alpha* len(in_stats))):
      		ucl = ucl - step_size
     		count = len([i for i in in_stats if i > ucl])
	
	#plotting
	if plotting:
		fig, ax = plt.subplots(figsize=(10, 7))
		lc = threshold_plot(ax, range(0, df.shape[0]), array(y_vals), ucl, 'b', 'r')
		ax.axhline(ucl, color='k', ls='--')
		ax.set_title(plot_title)
		ax.set_xlabel('Observation Number')
		ax.set_ylabel('MCUSUM statistic (Anomaly Score)')

		#saving plot
		if save != "":
			if save[-1] != '/':
				save += '/'
			plt.savefig(save + plot_title + '_FULL.png', dpi=300)
			
			#if there are a lot of entries, a smaller graph will also be saved that
			#shows only the end of the graph for more detail.
			if len(y_vals) > 10000:
				ax.set_xlim(9000)
				plt.savefig(save + plot_title + '_SMALL.png', dpi=300)
			
		
	#returns the mcusum statistics as a list and upper control limit
	return y_vals, ucl
