import pandas as pd
import numpy as np

def _univariate_t_test(data, n_in_control, alpha=0.05):
	"""
	Diagnostic test to be used as a followup to a multivariate signal
	t=(mu_new-mu-ref)/sqrt(sd_ref*(1/n_new+1/n_ref))
	Where ref represents in control data and new represents potetially out of control data

	Args:
		data: data used to calculate t-statistic
		n_in_control: the first ``n_in_control`` data points of ``data`` are sample1, the remaining are sample2
		alpha: significance level of the test (default = 5%)
	Returns:
		Calculated t-statistic for the test data[:n_in_control] vs data[n_in_control:]
	"""
	ref_data = data[:n_in_control]
	new_data = data[n_in_control:]
	n_ref = len(ref_data)
	n_new = len(new_data)

	mean_ref = ref_data.mean()
	sd_ref = ref_data.std()

	mean_new = new_data.mean()

	t = (mean_new - mean_ref) / np.sqrt(sd_ref * (1 / n_new + 1 / n_ref))
	return t


def _build_t_test_df(df, in_control_start, batch_size):
	"""
	Constructs a dataframe containing t-test statistics from multiple t-tests, the number
	of which is controlled by batch_size

	Args:
		df: dataframe which stores the data to be tested, with features as columns 
			and observations as rows.
		in_control_start: the starting index of the in control data
		batch_size: successive t-tests will be run on subsets of the potentially out of control
			segment of the dataset contained in ``df``. The size of each subset (aka batch) is 
			controlled by ``batch_size``
	Returns:
		Pandas DataFrame with floor(n_out_control/batch_size) rows where n_out_control=(len(df)-in_control_start).
		Each cell stores a test statistic.
	"""
	t_df = pd.DataFrame(columns=df.columns)
	# t_matrix = []
	m = 0
	for in_control in range(in_control_start, len(df), batch_size):
		row = {}
		for col in df.columns:
			end_index = in_control + batch_size
			t = _univariate_t_test(df[col][:end_index], in_control, alpha=0.05)
			row[col] = t
		t_df = t_df.append(row, ignore_index=True)
		m += 1
	t_df.index = list(range(in_control_start, len(df), batch_size))
	return t_df


def interpret_multivariate_signal(df,
								  stats_list,
								  ucl,
								  batch_size=5,
								  n_most_likely=5,
								  verbose=False):
	"""
	Designed to interpret an out of control signal from a multivariate control chart method. Used
	to identify the source of the signal by examining each individual feature using successive t-tests.
	Since this method relies on t-tests, it is ill-equipped to handle shifts in process variability.

	Args:
		df: dataframe which stores the data to be tested, with features as columns 
			and observations as rows.
		stats_list: list of calculated statistics from an MSPC method
		ucl: Upper Control Limit (returned by an MSPC method or chosen by the user)
		batch_size: successive t-tests will be run on subsets of the potentially out of control
			segment of the dataset contained in ``df``. The size of each subset (aka batch) is 
			controlled by ``batch_size``
		n_most_likely: if ``verbose=True`` then this controls how many of the most likely observations will
			be printed.
		verbose: if ``True``, prints the most likely cuplrit features
	Returns:
		A ranked list of features (as a Pandas series) sorted from highest to lowest average t-statistic ranking.
	"""
	num_before_signal = stats_list.index(next(filter(lambda i: i>ucl, stats_list))) 
# the number of observations which are in control (the observations before the signal)
	t_test_df = _build_t_test_df(df, num_before_signal, batch_size)

	ranks_srs = pd.Series(index=df.columns)
	ranks_srs[ranks_srs.index] = 0
	for index, row in t_test_df.iterrows():
		t_stats = row.sort_values(ascending=False) # bug? these should be magnitude-ranked
		ranks_srs[t_stats.index] += row.rank(ascending=False) # Rank each t-statistic
	ranks_srs = ranks_srs / len(
		t_test_df)  # Convert t statistics to an average ranking
	if verbose:
		print(
			'The most likely culprit features and average t-statistic ranking in decreasing order are:'
		)
		print(ranks_srs.sort_values()[:n_most_likely])
	return ranks_srs.sort_values()