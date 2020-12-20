import numpy as np
import pandas as pd
from numpy.linalg import inv as inverse
from .plotting import threshold_plot


def apply_mewma(df, num_in_control, lambd=0.1, alpha=.01, plotting=True, save="", plot_title="MEWMA"):
    """
    Args:
        df: multivariate dataset as Pandas DataFrame
        num_in_control: number of rows before anomalies begin
        lambd: smoothing parameter between 0 and 1; lower value = higher weightage to older observations; default is 0.1
        alpha: the percentage of false positives we want to allow, used for calculating the Upper Control Limit (default .01 or 1%)
		save: the directory to save the graphs to, if not changed from default, nothing will be saved
		plot_title: the title for the plot generated
    Returns:
        MEWMA statistic values
    """
    nrow, ncol = df.shape
    means = df.head(num_in_control).mean(axis=0)

    # create diff matrix
    v = np.zeros(shape=(nrow - 1, ncol))
    for i in range(nrow - 1):
        v[i] = df.iloc[i + 1] - df.iloc[i]

    # calculate vTv
    vtv = v.T @ v

    # calculate S matrix
    S = (1 / (2 * (nrow-1))) * (vtv)

    mx = df - means

    # calculate z
    z = np.zeros(shape=(nrow + 1, ncol))
    for i in range(nrow):
        z[i + 1] = lambd * mx.iloc[i] + (1 - lambd) * z[i]
    z = z[1:, :]

    t2 = []  # test statistic values
    for i in range(nrow):
        w = (lambd / (2 - lambd)) * (1 - (1 - lambd)**(2 * (i + 1)))
        inv = inverse(w * S)
        t2.append((z[i].T @ inv) @ z[i])
        
    #calculate UCL
	in_stats = t2[0:num_in_control]
  	ucl = max(in_stats)
 
  	count = len([i for i in in_stats if i > ucl]) 

  	while(count < (alpha* len(in_stats))):
      		ucl = ucl - step_size
     		count = len([i for i in in_stats if i > ucl])
	
	#plotting
	if plotting:
		fig, ax = plt.subplots(figsize=(10, 7))
		lc = threshold_plot(ax, range(0, df.shape[0]), array(t2), ucl, 'b', 'r')
		ax.axhline(ucl, color='k', ls='--')
		ax.set_title(plot_title)
		ax.set_xlabel('Observation Number')
		ax.set_ylabel('MEWMA statistic (Anomaly Score)')

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

    # return t2 values
    return t2


def pc_mewma(df, num_in_control, num_princ_comps, alpha=.01, plotting=True, save="", plot_title="PC_MEWMA"):
    """
    MEWMA on Principle Components
    Variables contained in `df` must have mean 0

    Args:
        df: multivariate dataset as Pandas DataFrame
        num_in_control: number of in control observations
        num_princ_comps: number of principle components to include
        alpha: the percentage of false positives we want to allow, used for calculating the Upper Control Limit (default .01 or 1%)
		save: the directory to save the graphs to, if not changed from default, nothing will be saved
		plot_title: the title for the plot generated
    Returns:
        MEWMA statistic values using PCA for dimensionality reduction
    """
    df = df-df.mean()
    pca = PCA(n_components=num_princ_comps)
    s = pca.fit(df)
    eigvec_mat = s.components_ # V transpose
    W_matrix = []  # since apply_mewma only takes a df, not observations as they happen
    for index, row in df.iterrows():
        W = eigvec_mat @ row
        W_matrix.append(W)
    W_df = pd.DataFrame(W_matrix)
    return apply_mewma(W_df,
                       num_in_control,
                       lambd=0.1,
                       alpha,
                       plotting,
                       save,
                       plot_title)
