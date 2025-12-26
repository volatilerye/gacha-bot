"""handle item model."""

import re
from fractions import Fraction
from typing import Collection, Self

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

LINE_SEPARATORS = "[\r\n]"
COMMA_PATTERN = "[,、，]"


class ProbabilityError(Exception):
    """errors about probability."""


class PreItemModel(BaseModel):
    """item model before handling."""

    model_config = ConfigDict(frozen=True)

    name: str
    prob_weight: Fraction = Field(ge=0, default_factory=Fraction)
    required: int = Field(ge=0, default=1)

    @staticmethod
    def build_all(text: str) -> list[Self]:
        """build all PreItemModel instances from string.

        Args:
            text (str): item properties text.

        Returns:
            list[ItemModel]: generated item models.
        """
        return [
            PreItemModel.build(text=unit)
            for unit in re.split(pattern=LINE_SEPARATORS, string=text)
            if len(unit) > 0
        ]

    @staticmethod
    def build(text: str) -> Self:
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
        raise AssertionError(f"Expected code to be unreachable, but got: {text}")


class ItemModel(BaseModel):
    """item model."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(default_factory=str)
    prob: Fraction = Field(default_factory=Fraction, ge=0, le=1)
    required: int = Field(ge=0, default=1)
    _quantity: int = PrivateAttr(default=0)

    @staticmethod
    def build_from_text(text: str) -> list[Self]:
        """build preprocessed item list from string.

        Args:
            text (str): item properties text.

        Args:
            list[PreItemModel]: preprocessed item models.
        """
        pre_models = PreItemModel.build_all(text)
        return ItemModel.build_from_pre_model(pre_models)

    @staticmethod
    def build_from_pre_model(items: Collection[PreItemModel]) -> list[Self]:
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
        if all(isinstance(item.prob_weight, int) for item in items):
            return [
                PreItemModel(
                    name=item.name,
                    prob=Fraction(item.prob_weight, sum_weight),
                    required=item.required,
                )
                for item in items
            ]

        if sum_weight > 1:
            raise ProbabilityError("the probability of all events is over 1.")

        return [
            PreItemModel(name=item.name, prob=item.prob_weight, required=item.required)
            for item in items
        ]

    def add_quantity(self) -> None:
        """add stock quantity."""
        self._quantity += 1

    def get_quantity(self) -> int:
        """add stock quantity."""
        return self._quantity
