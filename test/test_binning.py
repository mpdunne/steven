import pytest
import random
import numpy as np
import pandas as pd

from steven.binning import get_bin_indices_discrete, get_bin_indices_continuous


@pytest.fixture(scope='session')
def list_discrete():
    rng = random.Random(1337)
    items = ['dog'] * 50 + ['cat'] * 50 + ['aardvark'] * 200
    rng.shuffle(items)
    return items


@pytest.fixture(scope='session')
def list_continuous():
    rng = random.Random(1337)
    values = [x / 2 for x in range(100)] + [50 + x / 4 for x in range(200)]
    rng.shuffle(values)
    return values


@pytest.mark.parametrize("input_type", ["list", "tuple"])
def test_get_bin_indices_discrete_inputs_works(list_discrete, input_type):
    if input_type == "list":
        data = list_discrete
    elif input_type == "tuple":
        data = tuple(list_discrete)
    else:
        raise ValueError(input_type)

    bins = get_bin_indices_discrete(data)

    assert isinstance(bins, dict)
    all_indices = []
    for value, indices in bins.items():
        assert all(data[i] == value for i in indices)
        all_indices.extend(indices)

    assert sorted(all_indices) == list(range(len(list_discrete)))


def test_get_bin_indices_discrete_unhashable_list():
    bad_data = [[1], [2], [1]]
    with pytest.raises(TypeError):
        get_bin_indices_discrete(bad_data)


def test_get_bin_indices_discrete_non_flat_raises():
    bad_data = [[1, 2], 3, 4]
    with pytest.raises(TypeError):
        get_bin_indices_discrete(bad_data)


def test_get_bin_indices_discrete_non_sequence_raises():
    bad_data = {"a": 1, "b": 2}
    with pytest.raises(TypeError):
        get_bin_indices_discrete(bad_data)


def test_get_bin_indices_discrete_empty():
    bins = get_bin_indices_discrete([])
    assert isinstance(bins, dict)
    assert bins == {}


@pytest.mark.parametrize("input_type", ["list", "tuple", "np_array", "pd_series"])
def test_get_bin_indices_continuous_inputs_works(list_continuous, input_type):
    if input_type == "list":
        data = list_continuous
    elif input_type == "tuple":
        data = tuple(list_continuous)
    elif input_type == "np_array":
        data = np.array(list_continuous)
    elif input_type == "pd_series":
        data = pd.Series(list_continuous)
    else:
        raise ValueError(input_type)

    bins = get_bin_indices_continuous(data, n_bins=10)

    assert isinstance(bins, dict)
    all_indices = []
    for (l, r), indices in bins.items():
        if len(indices) == 0:
            continue
        pulled = np.asarray(data)[indices]
        assert (pulled >= l).all()
        assert (pulled <= r).all()
        all_indices.extend(indices)

    assert sorted(all_indices) == list(range(len(list_continuous)))


def test_get_bin_indices_continuous_non_numeric_raises():
    bad_data = ["apple", "banana", "carrot"]
    with pytest.raises(TypeError):
        get_bin_indices_continuous(bad_data, n_bins=5)


def test_get_bin_indices_continuous_non_flat_raises():
    bad_data = [[1.0, 2.0], 3.0, 4.0]
    with pytest.raises(TypeError):
        get_bin_indices_continuous(bad_data, n_bins=5)


def test_get_bin_indices_continuous_non_sequence_raises():
    bad_data = {"a": 1.0, "b": 2.0}
    with pytest.raises(ValueError):
        get_bin_indices_continuous(bad_data, n_bins=5)


def test_get_bin_indices_continuous_zero_range_raises():
    data = [1.0, 1.0, 1.0]
    with pytest.raises(ValueError):
        get_bin_indices_continuous(data, n_bins=3)


def test_get_bin_indices_continuous_empty_raises():
    with pytest.raises(ValueError):
        bins = get_bin_indices_continuous([], n_bins=5)
