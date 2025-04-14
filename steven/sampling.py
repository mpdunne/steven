import numpy as np
import random

from copy import deepcopy
from tqdm.auto import tqdm
from typing import Any, Hashable, List, Sequence, Union


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
