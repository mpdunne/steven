import pandas as pd
import pytest
import random

from steven.binning import bin_series_discrete, bin_series_continuous


@pytest.fixture(scope='session')
def series_continuous():
    rng = random.Random(1337)
    values = [x / 2 for x in range(100)] + [50 + x / 4 for x in range(200)]
    rng.shuffle(values)
    return pd.Series(values)


@pytest.fixture(scope='session')
def series_discrete():
    rng = random.Random(1337)
    items = ['dog'] * 50 + ['cat'] * 50 + ['aardvark'] * 200
    rng.shuffle(items)
    return pd.Series(items)


@pytest.mark.parametrize("bad_index", [True, False])
def test_bin_series_discrete(series_discrete, bad_index):
    if bad_index:
        series_discrete.index = series_discrete.index % 10
    values, value_groups = bin_series_discrete(series_discrete, return_values=True)
    for v, vg in zip(values, value_groups):
        assert (vg == v).all()
        assert len(vg) == (series_discrete == v).sum()


@pytest.mark.parametrize("bad_index", [True, False])
def test_bin_series_continuous(series_continuous, bad_index):
    if bad_index:
        series_continuous.index = series_continuous.index % 10
    bin_edges, bins = bin_series_continuous(series_continuous, n_bins=10, return_bin_edges=True)
    for (bin_l, bin_r), bin in zip(bin_edges, bins):
        assert ((bin_l <= bin) & (bin <= bin_r)).all()
    assert set([x for bin in bins for x in bin]) == set(series_continuous)
