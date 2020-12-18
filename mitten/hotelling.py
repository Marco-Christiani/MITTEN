import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyplot_themes as themes
import pandas as pd
from .plotting import threshold_plot

def hotelling_t2(df,
                 alpha,
                 n_in_control,
                 ucl=0,
                 plot_title="Hotelling's T2",
                 save=False,
                 save_dir=None,
                 verbose=True):
    """
    Args:
        df: multivariate dataset as Pandas DataFrame
        alpha: parameter to help determine Upper Control Limit (Not currently used)
        n_out_of_control: number of in control observations in dataset
        ucl: upper control limit
        plot_title: title of generated control chart plot
        save: if True, save plots
        save_dir: if set, directory to save plots in
        verbose: if True, plot data
    Returns:
        t^2 statistic values and control limit as tuple
    """
    # calculate covariance matrix
    S = df.cov()

    # finding x-mean vector for each row in dataframe
    s_inv = pd.DataFrame(np.linalg.pinv(S), S.columns, S.index)
    mean_vec = df.mean()  # 56 values
    t2_values = []
    for index, row in df.iterrows():
        diff = row.subtract(mean_vec)
        # calculate T2 values now: T2 = [x-mean]'S-1[x-mean]
        t2 = diff.transpose() @ s_inv
        t2 = t2@diff
        t2_values.append(t2)

    if verbose:
        # generate plot of t2 values
        themes.theme_ggplot2()
        fig, ax = plt.subplots(figsize=(10, 7))
        if ucl == 0:
            ax = sns.lineplot(data=t2_values, ax=ax)
        else:
            lc = threshold_plot(ax, range(0, len(t2_values)),
                                np.array(t2_values), h, 'b', 'r')
            ax.axhline(ucl, color='k', ls='--')
        ax.set_xlabel('Observation')
        ax.set_ylabel('T2 Value')
        ax.set_title(plot_title)
        if save:
            if save_dir:
                if save_dir[-1] != '/':
                    save_dir += '/'
                plt.savefig(save_dir + plot_title + '_FULL.png', dpi=300)
            else:
                raise Exception(
                    'Please provide a path to `save_dir` if `save` is set to `True`'
                )
        if len(t2_values) > 10000:
            ax.set_xlim(9000, len(t2_values))
            if save:
                if save_dir:
                    if save_dir[-1] != '/':
                        save_dir += '/'
                plt.savefig(save_dir + plot_title + '_SMALL.png', dpi=300)

    # return t2 values and upper control lim
    return t2_values, ucl

