import numpy as np

from numbers import Number
from typing import Dict, List, Sequence, Hashable, Tuple, Union


def get_bin_indices_discrete(values: Sequence[Hashable]) -> Dict[Hashable, List[int]]:
    """
    Organise a sequence or 1d-array/series into bins of discrete values.
    Returns the set of distinct values, the groups of values, and (optional) the indices for each group.

    :data: The input sequence. Must be hashable (numeric, float, etc.).
    """
    if not isinstance(values, (list, tuple)):
        raise TypeError("Input must be a list or tuple.")

    bins = {}
    for ix, value in enumerate(values):

        if not isinstance(value, Hashable):
            raise TypeError(f"Unhashable value encountered at position {ix}: {value}")

        if value not in bins:
            bins[value] = []

        bins[value].append(ix)

    return bins


def get_bin_indices_continuous(values: Union[Sequence[Number], np.ndarray],
                               n_bins: int,
                               bin_range: Union[Tuple[float, float], None] = None
                               ) -> Dict[Tuple[float, float], List[int]]:
    """
    Organise a sequence or 1d-array/series into n evenly sized bins, and return a dict of
    index sets keyed by bin edge tuples (l, r).

    :values: The input sequence. Must be hashable (numeric, float, etc.).
    :n_bins: The number of bins to use.
    :bin_range: The range in which to bin.

    """
    value_array = np.asarray(values)

    if value_array.ndim != 1:
        raise ValueError("Input data must be 1-dimensional.")

    if not np.issubdtype(value_array.dtype, np.number):
        raise TypeError("Input data must be numeric.")

    if bin_range is None:
        bin_min, bin_max = value_array.min(), value_array.max()
    else:
        bin_min, bin_max = bin_range

    if bin_min == bin_max:
        raise ValueError("bin_range must span a nonzero range.")

    bin_edges = np.linspace(bin_min, bin_max, n_bins + 1)

    bins = {}

    for i in range(n_bins):
        bin_l = bin_edges[i]
        bin_r = bin_edges[i + 1]

        if i < n_bins - 1:
            mask = (value_array >= bin_l) & (value_array < bin_r)
        else:
            mask = (value_array >= bin_l) & (value_array <= bin_r)

        bin_indices = np.where(mask)[0].tolist()
        bins[(bin_l, bin_r)] = bin_indices

    return bins
