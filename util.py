import math
from typing import Sequence


def convert_combination_to_index(n: int, combination: Sequence[int]) -> int:
    """Convert a combination to its lexicographic index.

    Args:
        n (int): The total number of items.
        k (int): The number of items in the combination.
        combination (Sequence[int]): The combination to convert.

    Returns:
        int: The lexicographic index of the combination.
    """
    k = len(combination)

    index = 0
    last_val = -1
    for i in range(k):
        current_val = combination[i]
        for j in range(last_val + 1, current_val):
            index += math.comb(n - j - 1, k - i - 1)
        last_val = current_val
    return index


def convert_index_to_combination(n: int, k: int, index: int) -> tuple[int, ...]:
    """Convert a lexicographic index to its combination.

    Args:
        n (int): The total number of items.
        k (int): The number of items in the combination.
        index (int): The lexicographic index to convert.

    Returns:
        Sequence[int]: The combination corresponding to the index.
    """
    combination: list[int] = []
    last_val = -1
    for i in range(k):
        for j in range(last_val + 1, n):
            count = math.comb(n - j - 1, k - i - 1)
            if index < count:
                combination.append(j)
                last_val = j
                break
            else:
                index -= count
    return tuple(combination)


def convert_prod_combination_to_index(
    n: Sequence[int], combination: Sequence[Sequence[int]]
) -> int:
    """Convert a product combination to its lexicographic index.

    Args:
        n (Sequence[int]): The total number of items for each group.
        combination (Sequence[Sequence[int]]): The product combination to convert.

    Returns:
        int: The lexicographic index of the product combination.
    """
    index = 0
    multiplier = 1
    for group_n, group_comb in zip(reversed(n), reversed(combination)):
        group_k = len(group_comb)
        group_index = convert_combination_to_index(group_n, group_comb)
        index += group_index * multiplier
        multiplier *= math.comb(group_n, group_k)
    return index


def convert_index_to_prod_combination(
    n: Sequence[int], k: Sequence[int], index: int
) -> tuple[tuple[int, ...], ...]:
    """Convert a lexicographic index to its product combination.

    Args:
        n (Sequence[int]): The total number of items for each group.
        k (Sequence[int]): The number of items in each group combination.
        index (int): The lexicographic index to convert.

    Returns:
        Sequence[Sequence[int]]: The product combination corresponding to the index.
    """
    combination: list[tuple[int, ...]] = []
    multiplier = math.prod(math.comb(ni, ki) for ni, ki in zip(n, k))
    for group_n, group_k in zip(n, k):
        multiplier //= math.comb(group_n, group_k)
        group_index = index // multiplier
        group_comb = convert_index_to_combination(group_n, group_k, group_index)
        combination.append(group_comb)
        index %= multiplier
    return tuple(combination)
