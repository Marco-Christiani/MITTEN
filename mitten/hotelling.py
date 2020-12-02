import scipy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyplot_themes as themes
from .plotting import threshold_plot

def hotelling_t2(df,
                 alpha,
                 n_out_of_control,
                 h=0,
                 plot_title="Hotelling's T2",
                 save=False,
                 save_dir=None,
                 verbose=True):
    """
    Args:
        df: multivariate dataset as Pandas DataFrame
        alpha: parameter to help determine Upper Control Limit
        n_out_of_control: number of out of control observations in dataset
        h:
        plot_title: title of generated control chart plot
        save: boolean indicating whether to save the generated plot
        save_dir: directory in which to save the plot, if saving
    Returns:

    """
    # calculate difference matrix
    S = df.cov()

    # finding x-mean vector for each row in dataframe
    s_inv = pd.DataFrame(np.linalg.pinv(S), S.columns, S.index)
    mean_vec = df.mean()  # 56 values
    t2_values = []
    for index, row in df.iterrows():
        diff = row.subtract(mean_vec)
        # calculate T2 values now: T2 = [x-mean]'S-1[x-mean]
        t2 = diff.transpose() @ (s_inv)
        t2 = t2 @ (diff)
        t2_values.append(t2)

    # calculate the upper control limits
    m = df.shape[0] - n_out_of_control  # n in-control observations
    p = df.shape[1]  # num dimensions
    n = 1  # sample group size (indiv observations)
    q = 2 * (m - 1) * (m - 1) / ((3 * m) - 4)  # for calculating beta distr

    # UNCOMMENT THESE THREE LINES FOR OLD VERSION
    df1 = p
    df2 = m - p
    coef = p * (m - 1) / (m - p)

    # NEW VERSION
    # df1 = p
    # df2 = (m*n) - m - p + 1
    # coef =

    

    alpha = .05
    f = scipy.stats.f.ppf(q=1 - alpha, dfn=df1, dfd=df2)
    ucl = coef * f
    # ucl

    # turn off plotting if verbose = false
    if not verbose:
        plt.ioff()

    # generate plot of t2 values    
    themes.theme_ggplot2()
    fig, ax = plt.subplots(figsize=(10, 7))
    # ax = sns.scatterplot(data=sim_df_copy['t2'])
    if h == 0:
        ax = sns.lineplot(data=t2_values, ax=ax)
    # ax.axhline(ucl, color='r')
    else:
        # ax.axhline(h, color='r')
        lc = threshold_plot(ax, range(0, len(t2_values)),
                            np.array(t2_values), h, 'b', 'r')
        ax.axhline(h, color='k', ls='--')
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

    plt.ion()

