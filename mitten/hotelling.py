import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .plotting import threshold_plot
from .helpers import calculate_ucl


def hotelling_t2(df, num_in_control, alpha=0, plotting=True, save='', plot_title='Hotellings T^2'):
	"""
	Args:
		df: multivariate dataset as Pandas DataFrame
		num_in_control: number of in control observations before the anomalies start
		alpha: the percentage of false positives we want to allow, used for calculating the Upper Control Limit
		save: the directory to save the graphs to, if not changed from default, nothing will be saved
		plot_title: the title for the plot generated
	Returns:
		Hotelling T^2 statistic values and a calculated UCL with approximately ``alpha`` false positive rate
	"""
	# calculate covariance matrix
	S = df.head(num_in_control).cov()

	# finding x-mean vector for each row in dataframe
	s_inv = pd.DataFrame(np.linalg.pinv(S), S.columns, S.index)
	mean_vec = df.head(num_in_control).mean()  # 56 values
	t2_values = []
	for index, row in df.iterrows():
		diff = row.subtract(mean_vec)
		# calculate T2 values now: T2 = [x-mean]'S-1[x-mean]
		t2 = diff.transpose() @ s_inv
		t2 = t2@diff
		t2_values.append(t2)
	   
	#calculate UCL
	in_stats = t2_values[0:num_in_control]
 
	ucl = calculate_ucl(in_stats, alpha)
	  
	#plotting
	if plotting:
		plt.style.use('ggplot')
		fig, ax = plt.subplots(figsize=(10, 7))
		lc = threshold_plot(ax, range(0, df.shape[0]), np.array(t2_values), ucl, 'b', 'r')
		ax.axhline(ucl, color='k', ls='--')
		ax.set_title(plot_title)
		ax.set_xlabel('Observation Number')
		ax.set_ylabel('Hotelling statistic (Anomaly Score)')

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

	# return Hotelling T^2 statistic values
	return t2_values, ucl
