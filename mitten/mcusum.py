import numpy as np
import matplotlib.pyplot as plt
from .plotting import threshold_plot
from numpy import matmul, sqrt, array

def mcusum(df,
           num_in_control,
           k,
           ucl,
           plot_title='MCUSUM',
           save=False,
           verbose=False,
           save_dir=None):
    """
    Based on Kent(2007) https://etd.ohiolink.edu/rws_etd/send_file/send?accession=kent1185558637&disposition=inline
            --> reference 17 : (Jackson 1985)
            --> reference 5  : (Crosier 1988)
    Args:
      df: multivariate dataset as Pandas DataFrame
      num_in_control: number of in control observations
      k: decay parameter
      ucl: Upper control limit
      plot_title: title of plot
      save: if true, saves plot image
      save_dir: if `save` is true, directory where plot image should be saved
      verbose: if true, plots results
    Returns:
        MCUSUM statistic values and upper control limit as tuple
    """
    a = df.head(num_in_control).mean(axis=0)  # mean vector of in control data
    cov_inv = np.linalg.inv(np.cov(df, rowvar=False,
                                   bias=True))  # covariance matrix inverted
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

    if verbose:  #plotting
        fig, ax = plt.subplots(figsize=(10, 7))
        lc = threshold_plot(ax, range(0, df.shape[0]), array(y_vals), ucl, 'b',
                            'r')
        ax.axhline(ucl, color='k', ls='--')
        ax.set_title(plot_title)
        ax.set_xlabel('Observation Number')
        ax.set_ylabel('MCUSUM statistic (Anomaly Score)')

        #saving plot
        if save:
            if save_dir:
                if save_dir[-1] != '/':
                    save_dir += '/'
                plt.savefig(save_dir + plot_title + '_FULL.png', dpi=300)
            else:
                raise Exception(
                    'Please provide a path to `save_dir` if `save` is set to `True`'
                )
        #if there are a lot of entries, a smaller graph will also be saved that
        #shows only the end of the graph for more detail.        
        if len(y_vals) > 10000:
            ax.set_xlim(9000)
            if save:
                if save_dir:
                    if save_dir[-1] != '/':
                        save_dir += '/'
                plt.savefig(save_dir + plot_title + '_SMALL.png', dpi=300)
    #returns the mcusum statistics as a list and the upper control limit provided
    return y_vals, ucl
