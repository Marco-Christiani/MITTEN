import numpy as np
import matplotlib.pyplot as plt
from .plotting import threshold_plot


def mcusum(df,
           num_normal,
           k,
           h,
           plot_title='MCUSUM',
           save=False,
           verbose=False,
           save_dir=None):
    """
    Based on Kent(2007) https://etd.ohiolink.edu/rws_etd/send_file/send?accession=kent1185558637&disposition=inline
            --> reference 17 : (Jackson 1985)
            --> reference 5  : (Crosier 1988)
    Args:
      df:
      num_normal:
      k:
      h:
      plot_title:
      save:
      verbose:
    Returns:

    """
    a = df.head(num_normal).mean(axis=0)  #mean vector of normal data
    cov_inv = np.linalg.inv(np.cov(df, rowvar=False,
                                   bias=True))  #covariance matrix inverted
    s_old = [0] * df.shape[1]

    y_vals = [0] * df.shape[0]

    len_list = []

    run_length = 0
    for n in range(0, df.shape[0]):
        # get current line
        x = df.iloc[n]

        # calculate Cn
        tf = np.array(s_old + x - a)
        c = np.matmul(np.matmul(tf, cov_inv), tf)
        c = np.sqrt(c)

        # calculate kv (vector k)
        kv = (k / c) * tf

        # print('c: ', c, 'k: ', k)
        # calculate new s
        if (c <= k):
            s_old = 0
            len_list.append(run_length)
            run_length = 0
        else:
            s_old = tf * (1 - (k / c))
            run_length += 1
            # calculate y (new c)
            tf = np.array(s_old + x - a)
            y = np.matmul(np.matmul(tf, cov_inv), tf)
            y = np.sqrt(y)
            y_vals[n] = y

        # classify anomaly from limit
        # if (y > h and verbose):
        # print('SIGNAL: ' ,round(y), ', ind: ' ,n)
        # else:
        # print('normal: ', round(y), ' ind: ' , n)
    if (len(len_list)):
        print('average run length: ', sum(len_list) / len(len_list))
    else:
        print('average run length: 1, (k is small)')
    fig, ax = plt.subplots(figsize=(10, 7))
    lc = threshold_plot(ax, range(0, df.shape[0]), np.array(y_vals), h, 'b',
                        'r')
    ax.axhline(h, color='k', ls='--')
    ax.set_title(plot_title)
    ax.set_xlabel('Observation Number')
    ax.set_ylabel('MCUSUM statistic (Anomaly Score)')

    if save:
        if save_dir:
            if save_dir[-1] != '/':
                save_dir += '/'
            plt.savefig(save_dir + plot_title + '_FULL.png', dpi=300)
        else:
            raise Exception(
                'Please provide a path to `save_dir` if `save` is set to `True`'
            )
    if len(y_vals) > 10000:
        ax.set_xlim(9000)
        if save:
            if save_dir:
                if save_dir[-1] != '/':
                    save_dir += '/'
            plt.savefig(save_dir + plot_title + '_SMALL.png', dpi=300)