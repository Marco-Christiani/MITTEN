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
                 save_dir=None):
    """
    Args:
        df:
        alpha:
        n_out_of_control:
        h:
        plot_title:
        save:
        save_dir:
    Returns:

    """
    # calculate difference matrix
    sim_df_diff = df.diff()
    sim_df_diff.drop(index=0, inplace=True)
    sim_df_diff = sim_df_diff.reset_index(drop=True)
    sim_df_diff.head()  # has 199 rows instead of 200

    # calculate sample covariance matrix 'S'
    sim_df_diff_trans = sim_df_diff.transpose()
    s_matrix = sim_df_diff_trans.dot(sim_df_diff)
    s_matrix = s_matrix.applymap(lambda x: x / (2 * 199))

    S = df.cov()

    # finding x-mean vector for each row in dataframe
    s_inv = pd.DataFrame(np.linalg.pinv(S), S.columns, S.index)
    mean_vec = df.mean()  # 56 values
    t2_values = []
    for index, row in df.iterrows():
        diff = row.subtract(mean_vec)
        # calculate T2 values now: T2 = [x-mean]'S-1[x-mean]
        t2 = diff.transpose().dot(s_inv)
        t2 = t2.dot(diff)
        t2_values.append(t2)

    # add the t2 list as column to the df
    sim_df_copy = df.copy(deep=True)
    sim_df_copy = sim_df_copy.assign(t2=t2_values)
    # sim_df_copy.head()

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

    # generate plot of t2 values
    themes.theme_ggplot2()
    fig, ax = plt.subplots(figsize=(10, 7))
    # ax = sns.scatterplot(data=sim_df_copy['t2'])
    if h == 0:
        ax = sns.lineplot(data=sim_df_copy['t2'], ax=ax)
    # ax.axhline(ucl, color='r')
    else:
        # ax.axhline(h, color='r')
        lc = threshold_plot(ax, range(0, len(sim_df_copy['t2'])),
                            np.array(sim_df_copy['t2']), h, 'b', 'r')
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
    if len(sim_df_copy['t2']) > 10000:
        ax.set_xlim(9000, len(sim_df_copy['t2']))
        if save:
            if save_dir:
                if save_dir[-1] != '/':
                    save_dir += '/'
            plt.savefig(save_dir + plot_title + '_SMALL.png', dpi=300)

