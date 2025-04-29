from typing import Hashable

import pandas as pd

from steven.binning import get_bin_indices_discrete, get_bin_indices_continuous
from steven.sampling import sample_buckets_evenly


def subset_series_evenly(series: pd.Series,
                         sample_size: int,
                         mode: str = 'continuous',
                         n_bins=100,
                         random_state: Hashable = None,
                         progress: bool = True):
    """
    Sample a series according to some chosen binning system.
    Returns a list of lists of indices.

    :series: The input series.
    :sample_size: The number of bins in which to put data.
    :mode: Whether to treat the data as 'continuous' or 'discrete'.
    :n_bins: The number of bins, if mode='continuous'
    :random_state: The random state or random seed to use.
    """
    values = series.values

    if mode == 'continuous':
        bins, bin_ixs = zip(*get_bin_indices_continuous(values, n_bins=n_bins).items())
    elif mode == 'discrete':
        bins, bin_ixs = zip(*get_bin_indices_discrete(values).items())
    else:
        raise ValueError('Mode must be either continuous or discrete.')

    bins_ixs_sampled = sample_buckets_evenly(bin_ixs, sample_size, random_state=random_state, progress=progress)

    series_sampled = series.iloc[bins_ixs_sampled]
    return series_sampled
