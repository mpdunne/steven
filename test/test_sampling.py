import copy
import numpy as np
import pandas as pd
import pytest
import random

from steven.sampling import sample_buckets_evenly, sample_weighted, sample_data_evenly


@pytest.fixture(scope='session')
def items():
    return [['a1', 'a2', 'a3', 'a4'], ['b1', 'b2', 'b3'], ['c1', 'c2'], ['d1']]


def test_sample_buckets_evenly_total_too_big(items):
    total_size = sum(len(x) for x in items)
    with pytest.raises(ValueError) as e:
        _ = sample_buckets_evenly(items, total=total_size + 1)
    assert 'too large' in str(e.value)


def test_sample_buckets_evenly_total_same_size_as_n_items(items):
    total_size = sum(len(x) for x in items)
    result = sample_buckets_evenly(items, total=total_size)
    assert len(result) == total_size
    flat_items = {item for bucket in items for item in bucket}
    assert set(result) == flat_items


def test_sample_buckets_evenly_basic_functionality(items):
    sample = sample_buckets_evenly(items, total=4)
    assert len(sample) == 4
    # Should have one item from each bucket.
    heads = [s[0] for s in sample]
    assert len(heads) == 4
    assert set(heads) == {'a', 'b', 'c', 'd'}


def test_sample_buckets_evenly_does_not_modify_original_data(items):
    items_copy = copy.deepcopy(items)
    _ = sample_buckets_evenly(items, total=5)
    assert items == items_copy


def test_sample_buckets_evenly_same_seed_gives_same_result(items):
    sample1 = sample_buckets_evenly(items, total=6, random_state=8675309)
    sample2 = sample_buckets_evenly(items, total=6, random_state=8675309)
    assert sample1 == sample2


def test_sample_buckets_evenly_bucket_empty(items):
    sample = sample_buckets_evenly(items, total=8, random_state=8675309)
    assert len(sample) == 8
    heads = [s[0] for s in sample]
    assert heads.count('a') >= 2
    assert heads.count('b') >= 2
    assert heads.count('c') >= 2
    assert heads.count('d') == 1


@pytest.mark.parametrize("bucket_type", [list, tuple])
@pytest.mark.parametrize("outer_type", [list, tuple])
def test_sample_buckets_evenly_accepts_various_sequence_types(items, bucket_type, outer_type):
    converted_items = outer_type([bucket_type(bucket) for bucket in items])

    total_size = sum(len(bucket) for bucket in converted_items)
    result = sample_buckets_evenly(converted_items, total=total_size, random_state=123)
    assert len(result) == total_size

    flat_items = {item for bucket in converted_items for item in bucket}
    assert set(result) == flat_items


@pytest.fixture(scope='session')
def weighted_items():
    return ['a', 'b', 'c', 'd']


@pytest.fixture(scope='session')
def weights():
    return [0.1, 0.2, 0.6, 0.1]


def test_sample_weighted_with_replacement_length(weighted_items, weights):
    sample = sample_weighted(weighted_items, weights, k=10, replace=True, random_state=8675309)
    assert len(sample) == 10


def test_sample_weighted_without_replacement_length(weighted_items, weights):
    sample = sample_weighted(weighted_items, weights, k=4, replace=False, random_state=8675309)
    assert len(sample) == 4
    assert set(sample) <= set(weighted_items)


def test_sample_weighted_without_replacement_no_duplicates(weighted_items, weights):
    sample = sample_weighted(weighted_items, weights, k=4, replace=False, random_state=8675309)
    assert len(sample) == len(set(sample))  # All unique


def test_sample_weighted_with_replacement_allows_duplicates(weighted_items, weights):
    sample = sample_weighted(weighted_items, weights, k=50, replace=True, random_state=8675309)
    duplicates = len(sample) != len(set(sample))
    assert duplicates  # Should allow duplicates when replace=True


def test_sample_weighted_raises_if_k_too_big_no_replace(weighted_items, weights):
    with pytest.raises(ValueError) as e:
        _ = sample_weighted(weighted_items, weights, k=5, replace=False)
    assert 'cannot be more' in str(e.value)


def test_sample_weighted_raises_if_weights_mismatch(weighted_items):
    wrong_weights = [0.3, 0.7]  # wrong length
    with pytest.raises(ValueError) as e:
        _ = sample_weighted(weighted_items, wrong_weights, k=2)
    assert 'must match' in str(e.value)


def test_sample_weighted_distribution_bias(weighted_items, weights):
    sample = sample_weighted(weighted_items, weights, k=10000, replace=True, random_state=8675309)
    counts = {item: sample.count(item) for item in weighted_items}
    assert counts['c'] > counts['a']
    assert counts['c'] > counts['b']
    assert counts['c'] > counts['d']


def test_sample_weighted_correct_items_only(weighted_items, weights):
    sample = sample_weighted(weighted_items, weights, k=100, replace=True, random_state=8675309)
    for item in sample:
        assert item in weighted_items


def test_sample_weighted_deterministic_given_seed(weighted_items, weights):
    sample1 = sample_weighted(weighted_items, weights, k=10, replace=True, random_state=8675309)
    sample2 = sample_weighted(weighted_items, weights, k=10, replace=True, random_state=8675309)
    assert sample1 == sample2


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
def test_sample_data_evenly_discrete(discrete_data, input_type, return_ixs):
    data, expected_type = make_data(input_type, discrete_data)

    result = sample_data_evenly(data, mode='discrete', sample_size=99, return_ixs=return_ixs)
    if return_ixs:
        result, _ = result

    assert isinstance(result, expected_type), f"Expected {expected_type}, got {type(result)}"

    result_list = list(result)
    assert result_list.count('cat') == 33
    assert result_list.count('dog') == 33
    assert result_list.count('aardvark') == 33

    result = sample_data_evenly(data, mode='discrete', sample_size=200, return_ixs=return_ixs)
    if return_ixs:
        result, _ = result

    assert isinstance(result, expected_type), f"Expected {expected_type}, got {type(result)}"

    result_list = list(result)
    assert result_list.count('cat') == 50
    assert result_list.count('dog') == 50
    assert result_list.count('aardvark') == 100


@pytest.mark.parametrize("input_type", ['series_clean', 'series_bad_index', 'array', 'list', 'tuple'])
@pytest.mark.parametrize("return_ixs", [True, False])
def test_sample_data_evenly_continuous(continuous_data, input_type, return_ixs):
    data, expected_type = make_data(input_type, continuous_data)

    result = sample_data_evenly(data, mode='continuous', n_bins=10, sample_size=100, return_ixs=return_ixs)
    if return_ixs:
        result, _ = result

    assert isinstance(result, expected_type), f"Expected {expected_type}, got {type(result)}"

    result_array = np.array(result)
    for i in range(10):
        bin_count = ((result_array >= 10 * i) & (result_array < 10 * (i + 1))).sum()
        assert bin_count == 10

    result = sample_data_evenly(data, mode='continuous', n_bins=10, sample_size=250, return_ixs=return_ixs)
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
def test_sample_data_evenly_bad_inputs(bad_input):
    with pytest.raises(TypeError):
        sample_data_evenly(bad_input, sample_size=10)


@pytest.mark.parametrize("mode", ['invalid', '', 'CONtinuous', None])
def test_sample_data_evenly_bad_mode(mode):
    data = np.arange(100)

    with pytest.raises(ValueError):
        sample_data_evenly(data, sample_size=10, mode=mode)


def test_sample_data_evenly_empty_input():
    # Empty list
    with pytest.raises(ValueError):
        sample_data_evenly([], sample_size=5)

    # Empty numpy array
    with pytest.raises(ValueError):
        sample_data_evenly(np.array([]), sample_size=5)