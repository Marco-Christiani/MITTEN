import numpy as np
import pandas as pd

def hotelling_t2(df,
                 num_in_control):
    """
    Args:
        df: multivariate dataset as Pandas DataFrame
        num_in_control: number of in control observations before the anomalies start
    Returns:
        t^2 statistic values
    """
    # calculate covariance matrix
    S = df.head(num_in_control).cov()

    # finding x-mean vector for each row in dataframe
    s_inv = pd.DataFrame(np.linalg.pinv(S), S.columns, S.index)
    mean_vec = df.head(num_in_control).mean()  # 56 values
    t2_values = []
    for index, row in df.iterrows():
        diff = row.subtract(mean_vec)
        # calculate T2 values now: T2 = [x-mean]'S-1[x-mean]
        t2 = diff.transpose() @ s_inv
        t2 = t2@diff
        t2_values.append(t2)

    # return t2 values
    return t2_values
