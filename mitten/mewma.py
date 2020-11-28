import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyplot_themes as themes
import pandas as pd
from .plotting import threshold_plot


def apply_mewma(df, lambd=0.1, h=0, plot_title="MEWMA", save=False, save_dir=None):
    """
    Args:
        df:
        lambd:
        h:
        plot_title:
        save:
        save_dir:
    Returns:

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
    S = np.zeros(shape=(ncol, ncol))
    for i in range(ncol):
        S[i] = (1 / (2 * (nrow - 1))) * (vtv[i])

    mx = df - means

    # calculate z
    z = np.zeros(shape=(nrow + 1, ncol))
    for i in range(nrow):
        z[i + 1] = lambd * mx.iloc[i] + (1 - lambd) * z[i]
    z = z[1:, :]

    t2 = []  # values
    for i in range(nrow):
        w = (lambd / (2 - lambd)) * (1 - (1 - lambd)**(2 * (i + 1)))
        inv = np.linalg.inv(w * S)
        t2.append((z[i].T @ inv) @ z[i])

    # calculate upper control limit
    ucl = 0  # idk this yet

    # plot values with UCL value
    themes.theme_ggplot2()
    fig, ax = plt.subplots(figsize=(10, 7))
    # ax = sns.scatterplot(data=t2)
    # ax = sns.lineplot(data=t2, ax=ax)
    if h == 0:
        ax = sns.lineplot(data=t2, ax=ax)
    else:
        lc = threshold_plot(ax, np.array(range(0, len(t2))), np.array(t2), h,
                            'b', 'r')
        ax.axhline(h, color='k', ls='--')
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
    if t2 > 10000:
        ax.set_xlim(9000, len(t2))
        if save:
            if save_dir:
                if save_dir[-1] != '/':
                    save_dir += '/'
            plt.savefig(save_dir + plot_title + '_SMALL.png', dpi=300)


def pc_mewma(df, num_in_control, num_princ_comps):
    """
    MEWMA on Principle Components
    Variables contained in `df` must have mean 0

    Args:
        df:
        num_in_control:
        num_princ_comps:
    Returns:

    """


    # pca = PCA()
    # princ_comp = pca.fit_transform(df)
    # plt.figure(figsize=(10,8))
    # ax = sns.lineplot(data=np.cumsum(pca.explained_variance_ratio_)[:11])
    # ax.set_xlabel('$k$')
    # ax.set_ylabel('Percent Explained')
    # ax.set_title('Percent of Variation Explained by PCs')

    in_control_df = pd.DataFrame(df.iloc[:num_in_control])
    # pc_mat = PCA(n_components = num_princ_comps).fit_transform(in_control_df) # matrix of PCs
    [_, S,
     Vt] = np.linalg.svd(in_control_df)  # ensures eigvecs are in correct order
    V = np.transpose(Vt)
    eigvec_mat = V[:, :
                   num_princ_comps]  # Only using the k leading right singular vectors
    # principalDf = pd.DataFrame(data = princ_comp, columns = ['pc 1', 'pc 2'])
    W_matrix = [
    ]  # since apply_mewma only takes a df, not observations as they happen
    for index, row in df.iterrows():
        W = np.transpose(eigvec_mat) @ row
        W_matrix.append(W)
    W_df = pd.DataFrame(W_matrix)
    apply_mewma(W_df,
                lambd=0.1,
                h=0,
                title=f'Principal Component MEWMA, k={num_princ_comps}',
                save=False)
