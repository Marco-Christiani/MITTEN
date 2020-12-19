import numpy as np
import pandas as pd
from numpy.linalg import inv as inverse


def apply_mewma(df, num_in_control, lambd=0.1):
    """
    Args:
        df: multivariate dataset as Pandas DataFrame
        num_in_control: number of rows before anomalies begin
        lambd: smoothing parameter between 0 and 1; lower value = higher weightage to older observations; default is 0.1
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

    # return t2 values
    return t2


def pc_mewma(df, num_in_control, num_princ_comps):
    """
    MEWMA on Principle Components
    Variables contained in `df` must have mean 0

    Args:
        df: multivariate dataset as Pandas DataFrame
        num_in_control: number of in control observations
        num_princ_comps: number of principle components to include
    Returns:
        MEWMA statistic values using PCA for dimensionality reduction
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
                       num_in_control,
                       lambd=0.1)
