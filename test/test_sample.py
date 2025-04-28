import pytest
import numpy as np

from steven.sampling import sample_evenly


@pytest.fixture(scope='session')
def items():
    return [['a1', 'a2', 'a3', 'a4'], ['b1', 'b2', 'b3'], ['c1', 'c2'], ['d1']]


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
