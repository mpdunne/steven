import pytest
import numpy as np
import pandas as pd
import random

from steven.sampling import sample_evenly, bin_series_discrete, bin_series_continuous, subset_series_evenly


@pytest.fixture(scope='session')
def items():
    return [['a1', 'a2', 'a3', 'a4'], ['b1', 'b2', 'b3'], ['c1', 'c2'], ['d1']]


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


def test_sample_evenly_total_too_big(items):
    total_size = np.sum([len(x) for x in items])
    with pytest.raises(ValueError) as e:
        _ = sample_evenly(items, total=total_size + 1)
    assert 'too large' in e.value.args[0]


def test_sample_evenly_total_same_size_as_n_items(items):
    total_size = int(np.sum([len(x) for x in items]))
    _ = sample_evenly(items, total=total_size)


def test_sample_evenly_works(items):
    sample = sample_evenly(items, total=4)
    assert set([s[0] for s in sample]) == {'a', 'b', 'c', 'd'}
    sample = sample_evenly(items, total=8)
    assert set([s[0] for s in sample]) == {'a', 'b', 'c', 'd', 'a', 'b', 'c', 'a'}


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


@pytest.mark.parametrize("bad_index", [True, False])
def test_subset_series_evenly_discrete(series_discrete, bad_index):
    if bad_index:
        series_discrete.index = series_discrete.index % 10

    result = subset_series_evenly(series_discrete, mode='discrete', sample_size=99)
    assert (result == 'cat').sum() == 33
    assert (result == 'dog').sum() == 33
    assert (result == 'aardvark').sum() == 33

    result = subset_series_evenly(series_discrete, mode='discrete', sample_size=200)
    assert (result == 'cat').sum() == 50
    assert (result == 'dog').sum() == 50
    assert (result == 'aardvark').sum() == 100


@pytest.mark.parametrize("bad_index", [True, False])
def test_subset_series_evenly_continuous(series_continuous, bad_index):
    if bad_index:
        series_continuous.index = series_continuous.index % 10

    result = subset_series_evenly(series_continuous, mode='continuous', n_bins=10, sample_size=100)
    for i in range(10):
        assert ((10 * i <= result) & (result < 10 * (i + 1))).sum() == 10

    result = subset_series_evenly(series_continuous, mode='continuous', n_bins=10, sample_size=250)
    for i in range(5):
        assert ((10 * i <= result) & (result < 10 * (i + 1))).sum() == 20
    for i in range(5, 10):
        assert ((10 * i <= result) & (result < 10 * (i + 1))).sum() == 30
