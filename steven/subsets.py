import numpy as np
import pandas as pd

from typing import Hashable, List, Union, Tuple

from steven.binning import get_bin_indices_discrete, get_bin_indices_continuous
from steven.sampling import sample_buckets_evenly


SequenceLike = Union[list, tuple, np.ndarray, pd.Series]


def subset_data_evenly(data: SequenceLike,
                       sample_size: int,
                       mode: str = 'continuous',
                       n_bins=100,
                       random_state: Hashable = None,
                       progress: bool = True,
                       return_ixs: bool = False) -> Union[SequenceLike, Tuple[SequenceLike, List[int]]]:
    """
    Sample a series according to some chosen binning system.
    Returns the subsetted data, and optionally a list of which indices have survived

    :data: The input data (list, tuple, 1-D numpy array, or Series).
    :sample_size: The number of items to sample across all bins.
    :mode: Whether to treat the data as 'continuous' or 'discrete'.
    :n_bins: The number of bins, if mode='continuous'
    :random_state: The random state or random seed to use.
    :return_ixs: Whether to return the list of chosen indices.
    """
    if isinstance(data, pd.Series):
        values = data.values
    elif isinstance(data, (np.ndarray, list, tuple)):
        values = data
    else:
        raise TypeError('Inputted data must be of type list, tuple, 1-D numpy array or series')

    if mode == 'continuous':
        _, bin_ixs = zip(*get_bin_indices_continuous(values, n_bins=n_bins).items())
    elif mode == 'discrete':
        _, bin_ixs = zip(*get_bin_indices_discrete(values).items())
    else:
        raise ValueError('Mode must be either continuous or discrete.')

    sampled_ixs = sample_buckets_evenly(bin_ixs, sample_size, random_state=random_state, progress=progress)

    if isinstance(data, pd.Series):
        data_sampled = data.iloc[sampled_ixs]
    elif isinstance(data, np.ndarray):
        data_sampled = data[sampled_ixs]
    elif isinstance(data, list):
        data_sampled = [data[i] for i in sampled_ixs]
    elif isinstance(data, tuple):
        data_sampled = tuple([data[i] for i in sampled_ixs])

    if return_ixs:
        return data_sampled, sampled_ixs
    else:
        return data_sampled
