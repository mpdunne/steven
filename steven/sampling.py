import numpy as np
import random
import pandas as pd

from copy import deepcopy
from tqdm.auto import tqdm
from typing import Any, Hashable, List, Sequence, Tuple, Union


def sample_evenly(buckets: List[List[Any]],
                  total: int,
                  seed: Hashable = None,
                  progress: bool = True):
    """
    Sample evenly across a given set of buckets, up to a target total amount.

    :param buckets: The buckets from which to sample.
    :param total: The total size of the desired sample.
    :param seed: The random seed
    :param progress: Whether to display progress bar
    :return: A list of items from the buckets.
    """
    result = []
    total_sampled = 0

    n_items = np.sum([len(item) for item in buckets])
    if total > n_items:
        raise ValueError(f'Requested total too large: {total} > {n_items}')

    rng = random.Random(seed)
    with tqdm(total=total, disable=(not progress)) as pbar:
        buckets_working = deepcopy(buckets)
        rng.shuffle(buckets_working)
        while True:
            for i in range(len(buckets_working)):
                bucket = buckets_working[i]
                if len(bucket) == 0:
                    continue
                else:
                    choice = rng.choice(bucket)
                    result.append(choice)
                    bucket.remove(choice)
                    total_sampled += 1
                    pbar.update()
                    if total_sampled == total:
                        return result


def bin_series_discrete(series: pd.Series,
                        return_values: bool = False) -> Union[List, Tuple[List, List]]:
    """
    Place the indices from a series into bins according to their values.
    Returns a list of lists of indices.

    :series: The input series.
    """
    values = [*set(series)]
    value_groups = [series[series == value] for value in values]

    if return_values:
        return values, value_groups
    else:
        return value_groups


def bin_series_continuous(series: pd.Series,
                          n_bins: int,
                          bin_range: List = None,
                          return_bin_edges: bool = False) -> Union[Tuple[List, List], List]:
    """
    Place the indices from a series into bins spanning n_bins equal continuous regions.
    Returns a list of lists of indices.
    :series: The input series.
    :n_bins: The number of bins in which to put data.
    """

    if bin_range is None:
        bin_min, bin_max = series.min(), series.max()
    else:
        bin_min, bin_max = bin_range

    bin_size = (bin_max - bin_min) / n_bins

    bin_edges, bins = [], []

    for i in range(n_bins):

        # Numerical errors mean sometimes the ends don't get included. Force them.
        bin_l = (bin_min + i * bin_size) if (i != 0) else bin_min
        bin_r = (bin_min + (i + 1) * bin_size) if (i != n_bins - 1) else bin_max
        bin_edges.append([bin_l, bin_r])

        if i < n_bins - 1:
            bins.append(series[(series >= bin_l) & (series < bin_r)])
        else:
            bins.append(series[(series >= bin_l) & (series <= bin_r)])

    if return_bin_edges:
        return bin_edges, bins
    else:
        return bins


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


def sample_weighted(items: Sequence[Any],
                    weights: Sequence[Union[float, int]],
                    k: int,
                    replace: bool = False,
                    random_state: Union[random.Random, Hashable] = None) -> List[int]:
    """
    Sample a sequence of items, with a weighting given to each item.

    :param items: A sequence of items.
    :param weights: A sequences of weights, one for each item.
    :param k: The number of items to sample.
    :param replace: Whether to sample with replacement.
    :param random_state: The random seed or random state to use.
    :return: A sampled sequence of items.
    """
    if len(weights) != len(items):
        raise ValueError('Number of weights must match number of items.')

    if k > len(items) and not replace:
        raise ValueError('Sample size cannot be more than len(items) if replace=False')

    rng = random_state if type(random_state) is random.Random else random.Random(random_state)

    if replace:
        items_chosen = rng.choices(items, weights=weights, k=k)
    else:
        ixs = [*range(len(items))]
        ixs_chosen = []
        for _ in range(k):
            ix = rng.choices(ixs, weights=[weights[i] for i in ixs])[0]
            ixs_chosen.append(ix)
            ixs.remove(ix)
        items_chosen = [items[i] for i in ixs_chosen]

    return items_chosen
