import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from typing import Any, Hashable, List, Union, Sequence, Tuple

from steven.seeds import Seedable, get_rng
from steven.binning import get_bin_indices_discrete, get_bin_indices_continuous


SequenceLike = Union[list, tuple, np.ndarray, pd.Series]


def sample_buckets_evenly(buckets: Sequence[Sequence[Any]],
                          total: int,
                          random_state: Seedable = None,
                          progress: bool = True) -> List[Any]:
    """
    Sample evenly across a given set of buckets, up to a target total amount. Buckets are
    treated independently, and items within each bucket are sampled randomly.

    :param buckets: A sequence of buckets (each a sequence of items) to sample from.
    :param total: The total number of items to sample.
    :param random_state: A random seed or random.Random instance.
    :param progress: Whether to display a progress bar.
    :return: A list of sampled items.
    """
    result = []
    total_sampled = 0

    n_items = sum(len(bucket) for bucket in buckets)
    if total > n_items:
        raise ValueError(f'Requested total too large: {total} > {n_items}')

    rng = get_rng(random_state)

    # The indices within each bucket. We do this to avoid changing the buckets themselves.
    bucket_inner_indices = [[*range(len(bucket))] for bucket in buckets]

    # The indices of the buckets. We shuffle these to make sure we don't bias to the earlier buckets.
    bucket_indices = [*range(len(buckets))]
    rng.shuffle(bucket_indices)

    with tqdm(total=total, disable=(not progress)) as pbar:

        while True:

            # Keep track of any buckets that we've emptied.
            empty_bucket_indices = []

            for i in bucket_indices:
                ixs = bucket_inner_indices[i]
                if not ixs:
                    empty_bucket_indices.append(i)
                    continue
                chosen_ix = rng.choice(ixs)
                result.append(buckets[i][chosen_ix])
                ixs.remove(chosen_ix)
                total_sampled += 1
                pbar.update(1)
                if total_sampled == total:
                    return result

            for bucket_ix in empty_bucket_indices:
                bucket_indices.remove(bucket_ix)


def sample_data_evenly(data: SequenceLike,
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


def sample_weighted(items: Sequence[Any],
                    weights: Sequence[Union[float, int]],
                    k: int,
                    replace: bool = False,
                    random_state: Seedable = None) -> List[Any]:
    """
    Sample a sequence of items, with a weighting given to each item.

    :param items: A sequence of items.
    :param weights: A sequence of weights, one for each item.
    :param k: The number of items to sample.
    :param replace: Whether to sample with replacement.
    :param random_state: The random seed or random state to use.
    :return: A sampled sequence of items.
    """
    if len(weights) != len(items):
        raise ValueError('Number of weights must match number of items.')

    if k > len(items) and not replace:
        raise ValueError('Sample size cannot be more than len(items) if replace=False')

    rng = get_rng(random_state)

    if replace:
        return rng.choices(items, weights=weights, k=k)
    else:
        ixs = list(range(len(items)))
        ixs_chosen = []
        for _ in range(k):
            ix = rng.choices(ixs, weights=[weights[i] for i in ixs])[0]
            ixs_chosen.append(ix)
            ixs.remove(ix)
        return [items[i] for i in ixs_chosen]