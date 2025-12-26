"""handle item model."""

from __future__ import annotations

import bisect
import random
import re
from fractions import Fraction
from typing import Collection, Generator, Self, overload, override

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

LINE_SEPARATORS = "[\r\n]"
COMMA_PATTERN = "[,、，]"


class ProbabilityError(Exception):
    """errors about probability."""


class FrozenBaseModel(BaseModel):
    """frozen pydantic base model."""

    model_config = ConfigDict(frozen=True)


class PreItemModel(FrozenBaseModel):
    """item model before handling."""

    name: str
    prob_weight: Fraction = Field(ge=0, default_factory=Fraction)
    required: int = Field(ge=0, default=1)

    @staticmethod
    def build_all(text: str) -> tuple[PreItemModel, ...]:
        """build all PreItemModel instances from string.

        Args:
            text (str): item properties text.

        Returns:
            list[ItemModel]: generated item models.
        """
        return tuple(
            PreItemModel.build(text=unit)
            for unit in re.split(pattern=LINE_SEPARATORS, string=text)
            if len(unit) > 0
        )

    @staticmethod
    def build(text: str) -> PreItemModel:
        """build PreItemModel instance from string.

        Args:
            text (str): item properties text.

        Returns:
            ItemModel: generated item model.
        """
        split_list = [
            unit
            for unit in re.split(pattern=COMMA_PATTERN, string=text, maxsplit=2)
            if len(unit) > 0
        ]

        match len(split_list):
            case 1:
                return PreItemModel(name=split_list[0])
            case 2:
                return PreItemModel(
                    name=split_list[0], prob_weight=Fraction(split_list[1])
                )
            case 3:
                return PreItemModel(
                    name=split_list[0],
                    prob_weight=Fraction(split_list[1]),
                    required=int(split_list[2]),
                )
            case _:
                pass

        raise AssertionError(f"Expected code to be unreachable, but got: {text}")


class ItemModel(FrozenBaseModel):
    """item model."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(default_factory=str)
    prob: Fraction = Field(default_factory=Fraction, ge=0, le=1)
    required: int = Field(default=1, ge=0)
    quantity: int = Field(default_factory=int)

    # use in calc_prob_dist
    is_grouped: bool = Field(default_factory=bool)  # False
    group_size: int = Field(default=1, gt=0)

    @staticmethod
    def build_from_text(text: str) -> tuple[ItemModel, ...]:
        """build preprocessed item list from string.

        Args:
            text (str): item properties text.

        Args:
            list[PreItemModel]: preprocessed item models.
        """
        pre_models = PreItemModel.build_all(text)
        return ItemModel.build_from_pre_model(pre_models)

    @staticmethod
    def build_from_pre_model(
        items: Collection[PreItemModel],
    ) -> tuple[ItemModel, ...]:
        """build preprocessed item list from PreItemModel instances.

        Args:
            items (Collection[PreItemModel]): item models.

        Args:
            list[PreItemModel]: preprocessed item models.
        """
        sum_weight = sum([item.prob_weight for item in items])
        if sum_weight == 0:
            raise ProbabilityError("probability of each item is 0.")

        # all weights are integer
        if all(item.prob_weight.denominator == 1 for item in items):
            return tuple(
                ItemModel(
                    name=item.name,
                    prob=Fraction(item.prob_weight, sum_weight),
                    required=item.required,
                )
                for item in items
            )

        if sum_weight > 1:
            raise ProbabilityError("the probability of all events is over 1.")

        return tuple(
            ItemModel(name=item.name, prob=item.prob_weight, required=item.required)
            for item in items
        )


class ItemTable(FrozenBaseModel):
    """handle an item table."""

    items: tuple[ItemModel, ...]

    @staticmethod
    def loads(text: str) -> ItemTable:
        """load an item table from string.

        Args:
            text (str): item properties text.

        Args:
            ItemTable: generated item table.
        """
        return ItemTable(items=ItemModel.build_from_text(text))

    def __len__(self) -> int:
        return len(self.items)

    @overload
    def __getitem__(self, index: int) -> ItemModel: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[ItemModel, ...]: ...

    def __getitem__(self, index: int | slice) -> ItemModel | tuple[ItemModel, ...]:
        return self.items[index]

    @override
    def __iter__(self) -> Generator[ItemModel, None, None]:
        yield from self.items

    def add_quantity(self, index: int) -> Self:
        """add item quantity.

        Args:
            index (int): index of the item to add quantity to.

        Returns:
            Self: a new instance with the updated item quantity.
        """
        return self.__class__(
            items=tuple(
                item.model_copy(
                    update={"quantity": item.quantity + 1} if i == index else {}
                )
                for i, item in enumerate(self)
            )
        )

    def reset_quantities(self) -> Self:
        """reset quantities of all items to zero.

        Returns:
            Self: a new instance with all item quantities reset to zero.
        """
        return self.__class__(
            items=tuple(item.model_copy(update={"quantity": 0}) for item in self)
        )

    def run_monte_carlo(self, attempts: int = 1000) -> list[int]:
        """run monte carlo simulation.

        Args:
            attempts (int, optional): number of attempts. Defaults to 1000.

        Returns:
            list[int]: list of gacha counts for each attempt.
        """
        attempt_count: int = 0
        gacha_count: int = 0

        results: list[int] = []

        cumulative_prob = [
            sum(item.prob for item in self[: i + 1]) for i in range(len(self))
        ]

        while attempt_count < attempts:
            gacha_table: ItemTable = self.reset_quantities()
            gacha_count = 0

            while True:
                index = bisect.bisect_right(cumulative_prob, random.random())
                if index < len(cumulative_prob):
                    gacha_table = gacha_table.add_quantity(index=index)

                gacha_count += 1
                if all(item.quantity >= item.required for item in gacha_table):
                    results.append(gacha_count)
                    attempt_count += 1
                    break

        return results

    def _optimize(self) -> Self:
        """optimize markov process to reduce number of state.

        Returns:
            Self: optimized item table.
        """
        # remove items with required == 0
        reduced_items = list(item for item in self if item.required > 0)

        # group items with the same appearance probability
        # and required quantity together
        for i, former in enumerate(reduced_items):
            if former.required == 0:
                continue
            for j, latter in enumerate(reduced_items[i + 1 :], start=i + 1):
                if former.prob == latter.prob and former.required == latter.required:
                    reduced_items[i] = reduced_items[i].model_copy(
                        update={
                            "is_grouped": True,
                            "group_size": reduced_items[i].group_size
                            + reduced_items[j].group_size,
                        }
                    )
                    reduced_items[j] = reduced_items[j].model_copy(
                        update={"required": 0}
                    )

        return self.__class__(
            items=tuple(item for item in reduced_items if item.required > 0)
        )

    def calc_prob_dist(self) -> np.ndarray:
        return np.array([1])
