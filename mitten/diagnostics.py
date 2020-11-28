import pandas as pd
import numpy as np

def univariate_t_test(data, n_in_control, alpha=0.05):
    """
    Diagnostic test to be used as a followup to a multivariate signal
    t=(mu_new-mu-ref)/sqrt(sd_ref*(1/n_new+1/n_ref))
    Where ref represents in control data and new represents potetially out of control data
    """
    ref_data = data[:n_in_control]
    new_data = data[n_in_control:]
    n_ref = len(ref_data)
    n_new = len(new_data)

    mean_ref = ref_data.mean()
    sd_ref = ref_data.std()

    mean_new = new_data.mean()

    t = (mean_new - mean_ref) / np.sqrt(sd_ref * (1 / n_new + 1 / n_ref))
    return t


def build_t_test_df(df, in_control_start, batch_size):
    t_df = pd.DataFrame(columns=df.columns)
    # t_matrix = []
    m = 0
    for in_control in range(in_control_start, len(df), batch_size):
        row = {}
        for col in df.columns:
            end_index = in_control + batch_size
            t = univariate_t_test(df[col][:end_index], in_control, alpha=0.05)
            row[col] = t
        # row = pd.DataFrame(row)
        t_df = t_df.append(row, ignore_index=True)
        # print(pd.DataFrame(row))
        m += 1
    t_df.index = list(range(in_control_start, len(df), batch_size))
    return t_df


def interpret_multivariate_signal(df,
                                  n_in_control,
                                  batch_size=5,
                                  n_most_likely=5):
    t_test_df = build_t_test_df(df, n_in_control, batch_size)

    ranks_srs = pd.Series(index=df.columns)
    ranks_srs[ranks_srs.index] = 0
    for index, row in t_test_df.iterrows():
        t_stats = row.sort_values(ascending=False)
        # print(t_stats[:5])
        ranks_srs[t_stats.index] += row.rank(ascending=False)
    ranks_srs = ranks_srs / len(
        t_test_df)  # Convert t statistics to an average ranking
    print(
        'The most likely culprit features and average t-statistic ranking in decreasing order are:'
    )
    print(ranks_srs.sort_values()[:n_most_likely])