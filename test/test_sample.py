import copy
import pytest

from steven.sampling import sample_evenly, sample_weighted


@pytest.fixture(scope='session')
def items():
    return [['a1', 'a2', 'a3', 'a4'], ['b1', 'b2', 'b3'], ['c1', 'c2'], ['d1']]


def test_sample_evenly_total_too_big(items):
    total_size = sum(len(x) for x in items)
    with pytest.raises(ValueError) as e:
        _ = sample_evenly(items, total=total_size + 1)
    assert 'too large' in str(e.value)


def test_sample_evenly_total_same_size_as_n_items(items):
    total_size = sum(len(x) for x in items)
    result = sample_evenly(items, total=total_size)
    assert len(result) == total_size
    flat_items = {item for bucket in items for item in bucket}
    assert set(result) == flat_items


def test_sample_evenly_basic_functionality(items):
    sample = sample_evenly(items, total=4)
    assert len(sample) == 4
    # Should have one item from each bucket.
    heads = [s[0] for s in sample]
    assert len(heads) == 4
    assert set(heads) == {'a', 'b', 'c', 'd'}


def test_sample_evenly_does_not_modify_original_data(items):
    items_copy = copy.deepcopy(items)
    _ = sample_evenly(items, total=5)
    assert items == items_copy


def test_sample_evenly_same_seed_gives_same_result(items):
    sample1 = sample_evenly(items, total=6, random_state=8675309)
    sample2 = sample_evenly(items, total=6, random_state=8675309)
    assert sample1 == sample2


def test_sample_evenly_bucket_empty(items):
    sample = sample_evenly(items, total=8, random_state=8675309)
    assert len(sample) == 8
    heads = [s[0] for s in sample]
    assert heads.count('a') >= 2
    assert heads.count('b') >= 2
    assert heads.count('c') >= 2
    assert heads.count('d') == 1


@pytest.mark.parametrize("bucket_type", [list, tuple])
@pytest.mark.parametrize("outer_type", [list, tuple])
def test_sample_evenly_accepts_various_sequence_types(items, bucket_type, outer_type):
    converted_items = outer_type([bucket_type(bucket) for bucket in items])

    total_size = sum(len(bucket) for bucket in converted_items)
    result = sample_evenly(converted_items, total=total_size, random_state=123)
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
