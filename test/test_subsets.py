import numpy as np
import pandas as pd
import pytest
import random

from steven.subsets import subset_data_evenly


@pytest.fixture(scope='session')
def discrete_data():
    rng = random.Random(1337)
    items = ['dog'] * 50 + ['cat'] * 50 + ['aardvark'] * 200
    rng.shuffle(items)
    return items


@pytest.fixture(scope='session')
def continuous_data():
    rng = random.Random(1337)
    values = [x / 2 for x in range(100)] + [50 + x / 4 for x in range(200)]
    rng.shuffle(values)
    return values


def make_data(input_type: str, base):
    """Helper to generate input data and expected output type."""

    if input_type == 'series_clean':
        return pd.Series(base), pd.Series

    elif input_type == 'series_bad_index':
        s = pd.Series(base)
        s.index = s.index % 10
        return s, pd.Series

    elif input_type == 'array':
        return np.array(base), np.ndarray

    elif input_type == 'list':
        return list(base), list

    elif input_type == 'tuple':
        return tuple(base), tuple

    else:
        raise ValueError(f"Unknown input type: {input_type}")


@pytest.mark.parametrize("input_type", ['series_clean', 'series_bad_index', 'array', 'list', 'tuple'])
@pytest.mark.parametrize("return_ixs", [True, False])
def test_subset_data_evenly_discrete(discrete_data, input_type, return_ixs):
    data, expected_type = make_data(input_type, discrete_data)

    result = subset_data_evenly(data, mode='discrete', sample_size=99, return_ixs=return_ixs)
    if return_ixs:
        result, _ = result

    assert isinstance(result, expected_type), f"Expected {expected_type}, got {type(result)}"

    result_list = list(result)
    assert result_list.count('cat') == 33
    assert result_list.count('dog') == 33
    assert result_list.count('aardvark') == 33

    result = subset_data_evenly(data, mode='discrete', sample_size=200, return_ixs=return_ixs)
    if return_ixs:
        result, _ = result

    assert isinstance(result, expected_type), f"Expected {expected_type}, got {type(result)}"

    result_list = list(result)
    assert result_list.count('cat') == 50
    assert result_list.count('dog') == 50
    assert result_list.count('aardvark') == 100


@pytest.mark.parametrize("input_type", ['series_clean', 'series_bad_index', 'array', 'list', 'tuple'])
@pytest.mark.parametrize("return_ixs", [True, False])
def test_subset_data_evenly_continuous(continuous_data, input_type, return_ixs):
    data, expected_type = make_data(input_type, continuous_data)

    result = subset_data_evenly(data, mode='continuous', n_bins=10, sample_size=100, return_ixs=return_ixs)
    if return_ixs:
        result, _ = result

    assert isinstance(result, expected_type), f"Expected {expected_type}, got {type(result)}"

    result_array = np.array(result)
    for i in range(10):
        bin_count = ((result_array >= 10 * i) & (result_array < 10 * (i + 1))).sum()
        assert bin_count == 10

    result = subset_data_evenly(data, mode='continuous', n_bins=10, sample_size=250, return_ixs=return_ixs)
    if return_ixs:
        result, _ = result

    assert isinstance(result, expected_type), f"Expected {expected_type}, got {type(result)}"

    result_array = np.array(result)
    for i in range(5):
        bin_count = ((result_array >= 10 * i) & (result_array < 10 * (i + 1))).sum()
        assert bin_count == 20

    for i in range(5, 10):
        bin_count = ((result_array >= 10 * i) & (result_array < 10 * (i + 1))).sum()
        assert bin_count == 30


@pytest.mark.parametrize("bad_input", [
    {'a': 1, 'b': 2},
    {1, 2, 3},
    123,
    3.14,
    "hello",
])
def test_subset_data_evenly_bad_inputs(bad_input):
    with pytest.raises(TypeError):
        subset_data_evenly(bad_input, sample_size=10)


@pytest.mark.parametrize("mode", ['invalid', '', 'CONtinuous', None])
def test_subset_data_evenly_bad_mode(mode):
    data = np.arange(100)

    with pytest.raises(ValueError):
        subset_data_evenly(data, sample_size=10, mode=mode)


def test_subset_data_evenly_empty_input():
    # Empty list
    with pytest.raises(ValueError):
        subset_data_evenly([], sample_size=5)

    # Empty numpy array
    with pytest.raises(ValueError):
        subset_data_evenly(np.array([]), sample_size=5)