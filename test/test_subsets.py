import pandas as pd
import pytest
import random

from steven.subsets import subset_series_evenly


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
