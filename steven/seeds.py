import random

from typing import Hashable, Union

Seedable = Union[random.Random, Hashable]


def get_rng(random_state: Seedable) -> random.Random:
    """
    Obtain a random.Random instance from a Seedable type.

    :param random_state: Either a Hashable object or a random.Random instance.
    :return: A random.Random instance.
    """

    if isinstance(random_state, random.Random):
        return random_state

    return random.Random(random_state)
