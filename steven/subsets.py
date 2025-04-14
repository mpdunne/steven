from typing import Hashable

import pandas as pd

from steven.binning import bin_series_continuous, bin_series_discrete
from steven.sampling import sample_evenly

i


def subset_series_evenly(series: pd.Series,
                         sample_size: int,
                         mode: str = 'continuous',
                         n_bins=100,
                         random_seed: Hashable = None,
                         progress: bool = True):
    """
    Sample a series according to some chosen binning system.
    Returns a list of lists of indices.

    :series: The input series.
    :sample_size: The number of bins in which to put data.
    :mode: Whether to treat the data as 'continuous' or 'discrete'.
    :n_bins: The number of bins, if mode='continuous'
    :random_seed: The random seed to use.
    """
    series_reindexed = series.reset_index(drop=True, inplace=False)

    if mode == 'continuous':
        value_bins = bin_series_continuous(series_reindexed, n_bins=n_bins)
    elif mode == 'discrete':
        value_bins = bin_series_discrete(series_reindexed)
    else:
        raise ValueError('Mode must be either continuous or discrete.')

    value_bin_ixs = [[*x.index] for x in value_bins]
    value_bins_ixs_sampled = sample_evenly(value_bin_ixs, sample_size, seed=random_seed, progress=progress)

    series_sampled = series.iloc[value_bins_ixs_sampled]
    series_sampled.index = series.index[value_bins_ixs_sampled]
    return series_sampled