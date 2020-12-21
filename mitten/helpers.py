def calculate_ucl(in_control_stats, false_positive_rate, step_size=0.01):
	"""
	Given a target false positive rate, this method will automatically calculate
	an Upper Control Limit (UCL). This method should likely only be run with larger
	datasets for preliminary testing purposes.

	Args:
		in_control_stats: an array-like list of calulated statistic values (i.e. 
			from an MSPC method)
		false_positive_rate: the desired false positive rate
		step_size: the amount the UCL will be lowered each iteration until 
			``false_positive_rate`` is reached. Adjusting this will alter the number
			of iterations for convergence and might slightly decrease accuracy. 
	"""
	ucl = max(in_control_stats)
	ucl += step_size
	count = 0

	while(count < (false_positive_rate*len(in_control_stats))):
		ucl = ucl - step_size # lower UCL by step_size
		# count number of statistics above current ucl
		count = len([i for i in in_control_stats if i > ucl])
	return ucl