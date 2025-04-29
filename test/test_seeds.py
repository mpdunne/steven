import random
from steven.seeds import get_rng

n = 1_000_000_000


def test_get_rng_from_seed_int():
    seed = 8675309
    rng = get_rng(seed)
    assert isinstance(rng, random.Random)
    assert rng.randint(0, n) == random.Random(seed).randint(0, n)


def test_get_rng_from_seed_str():
    seed = "chicken"
    rng = get_rng(seed)
    assert isinstance(rng, random.Random)
    assert rng.randint(0, n) == random.Random(seed).randint(0, n)


def test_get_rng_from_existing_random_instance_same_object():
    seed = 5318008
    rng = random.Random(seed)
    new_rng = get_rng(rng)
    assert new_rng is rng


def test_get_rng_from_existing_random_instance_same_result():
    seed = 5318008
    rng = random.Random(seed)
    rng_result = rng.randint(0, n)
    new_rng = get_rng(random.Random(seed))
    new_rng_result = new_rng.randint(0, n)
    assert rng_result == new_rng_result


def test_get_rng_different_seeds_give_different_rngs():
    rng1 = get_rng(1)
    rng2 = get_rng(2)
    assert rng1.randint(0, 10000) != rng2.randint(0, n)


def test_get_rng_same_seed_gives_same_sequence():
    seed = 8675309
    rng1 = get_rng(seed)
    rng2 = get_rng(seed)
    seq1 = [rng1.randint(0, n) for _ in range(5)]
    seq2 = [rng2.randint(0, n) for _ in range(5)]
    assert seq1 == seq2
