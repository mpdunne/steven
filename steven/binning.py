from typing import Union, List, Tuple

import pandas as pd


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