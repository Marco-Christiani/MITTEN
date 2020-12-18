import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyplot_themes as themes
import pandas as pd
from .plotting import threshold_plot
from numpy.linalg import inv as inverse


def apply_mewma(df, lambd=0.1, ucl=0, plot_title="MEWMA", save=False, save_dir=None, verbose=True):
    """
    Args:
        df: multivariate dataset as Pandas DataFrame
        lambd: smoothing parameter between 0 and 1; lower value = higher weightage to older observations; default is 0.1
        ucl: upper control limit
        plot_title: title of generated control chart plot
        save: if True, save plots
        save_dir: if set, directory to save plots in
        verbose: if True, plot data
    Returns:
        MEWMA statistic values and control limit as tuple
    """
    nrow, ncol = df.shape
    means = df.mean(axis=0)

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

    if verbose:
        # plot values with UCL value
        themes.theme_ggplot2()
        fig, ax = plt.subplots(figsize=(10, 7))

        if ucl == 0:
            ax = sns.lineplot(data=t2, ax=ax)
        else:
            lc = threshold_plot(ax, np.array(range(0, len(t2))), np.array(t2), ucl,
                                'b', 'r')
            ax.axhline(ucl, color='k', ls='--')
        # ax.axhline(ucl, color='r')
        ax.set_xlabel('Observation')
        ax.set_ylabel('MEWMA Statistic')
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
        if len(t2) > 10000:
            ax.set_xlim(9000, len(t2))
            if save:
                if save_dir:
                    if save_dir[-1] != '/':
                        save_dir += '/'
                plt.savefig(save_dir + plot_title + '_SMALL.png', dpi=300)

    # return t2 values and upper control lim
    return t2, ucl


def pc_mewma(df, num_in_control, num_princ_comps, ucl=0, verbose=False):
    """
    MEWMA on Principle Components
    Variables contained in `df` must have mean 0

    Args:
        df: multivariate dataset as Pandas DataFrame
        num_in_control: number of in control observations
        num_princ_comps: number of principle components to include
        ucl: upper control limit
    Returns:
        MEWMA statistic values using PCA for dimensionality reduction and control limit as tuple
    """
    in_control_df = pd.DataFrame(df.iloc[:num_in_control])
    [_, S, Vt] = np.linalg.svd(in_control_df)  # ensures eigvecs are in correct order
    V = np.transpose(Vt)
    eigvec_mat = V[:, :num_princ_comps]  # Only using the k leading right singular vectors
    W_matrix = []  # since apply_mewma only takes a df, not observations as they happen
    for index, row in df.iterrows():
        W = np.transpose(eigvec_mat) @ row
        W_matrix.append(W)
    W_df = pd.DataFrame(W_matrix)
    return apply_mewma(W_df,
                       lambd=0.1,
                       ucl=ucl,
                       plot_title=f'Principal Component MEWMA, k={num_princ_comps}',
                       verbose=verbose)
