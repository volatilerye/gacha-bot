"""handle item model."""

from __future__ import annotations

import math
import operator
import re
from fractions import Fraction
from functools import reduce
from typing import Collection, Generator, Self, overload, override

# from itertools import
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator
from timeout_decorator import timeout

from util import convert_index_to_prod_combination, convert_prod_combination_to_index

LINE_SEPARATORS = "[\r\n]"
COMMA_PATTERN = "[,、，]"


class ProbabilityValidationError(Exception):
    """errors about probability field."""


class RequiredValidationError(Exception):
    """errors about required field."""


class DuplicatesValidationError(Exception):
    """errors about Duplicates field."""


class FrozenBaseModel(BaseModel):
    """frozen pydantic base model."""

    model_config = ConfigDict(frozen=True)


class PreItemModel(FrozenBaseModel):
    """item model before handling."""

    name: str
    prob_weight: Fraction
    required: int = Field(default=1)
    duplicates: int = Field(default=1)

    @field_validator("prob_weight")
    @classmethod
    def validate_prob_weight(cls, v: Fraction) -> Fraction:
        """validate proba_weight field."""
        if not (0 <= v and (v.denominator == 1 or 0 <= v <= 1)):
            raise ProbabilityValidationError(
                f"不正な確率（または確率の整数比）です: {v}"
            )
        return v

    @field_validator("required")
    @classmethod
    def validate_required(cls, v: int) -> int:
        """validate required field."""
        if v < 0:
            raise RequiredValidationError(f"不正な必要数です: {v}")
        return v

    @field_validator("duplicates")
    @classmethod
    def validate_duplicates(cls, v: int) -> int:
        """validate duplicates field."""
        if v < 0:
            raise DuplicatesValidationError(f"不正な重複数です: {v}")
        return v

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
            for unit in re.split(pattern=COMMA_PATTERN, string=text, maxsplit=3)
            if len(unit) > 0
        ]

        match len(split_list):
            case 1:
                raise ProbabilityValidationError(
                    f"確率（または確率の整数比）が指定されていません: {text}"
                )
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
            case 4:
                return PreItemModel(
                    name=split_list[0],
                    prob_weight=Fraction(split_list[1]),
                    required=int(split_list[2]),
                    duplicates=int(split_list[3]),
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

    # use in calc_prob_dist
    quantity_state: tuple[int, ...]

    @property
    def group_size(self) -> int:
        """get group size."""
        return sum(self.quantity_state)

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
        if any(item.prob_weight == 0 for item in items):
            raise ProbabilityValidationError(
                "確率が0のアイテムが存在します (アイテム収集が不可能です）"
            )

        sum_weight = sum([item.prob_weight * item.duplicates for item in items])

        # all weights are integer
        if all(item.prob_weight.denominator == 1 for item in items):
            return tuple(
                ItemModel(
                    name=item.name,
                    prob=Fraction(item.prob_weight, sum_weight),
                    required=item.required,
                    quantity_state=tuple([item.duplicates] + [0] * item.required),
                )
                for item in items
            )

        if sum_weight > 1:
            raise ProbabilityValidationError("全事象の確率が1を超えています")

        return tuple(
            ItemModel(
                name=item.name,
                prob=item.prob_weight,
                required=item.required,
                quantity_state=tuple([item.duplicates] + [0] * item.required),
            )
            for item in items
        )


class ItemTable(FrozenBaseModel):
    """handle an item table."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    items: tuple[ItemModel, ...]

    cache_mat: np.ndarray = Field(default=np.array([]))
    cache_mat_inv: np.ndarray = Field(default=np.array([]))

    @property
    def matrix_size(self) -> int:
        """calculate the probability matrix size.

        Returns:
            int: matrix size.
        """
        n_list = [item.required + item.group_size for item in self]
        k_list = [item.required for item in self]

        return reduce(operator.mul, [math.comb(n, k) for n, k in zip(n_list, k_list)])

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

    def optimize(self) -> Self:
        """optimize markov process to reduce number of state.

        Returns:
            Self: optimized item table.
        """
        # remove items with required == 0
        reduced_items = list(item for item in self if item.required > 0)

        # group items with the same appearance probability
        # and required quantity together
        for i, former in enumerate(reduced_items):
            if former.quantity_state == tuple():
                continue
            for j, latter in enumerate(reduced_items[i + 1 :], start=i + 1):
                if (
                    former.prob == latter.prob
                    and former.quantity_state == latter.quantity_state
                ):
                    reduced_items[i] = reduced_items[i].model_copy(
                        update={
                            "quantity_state": tuple(
                                f_qs + l_qs
                                for f_qs, l_qs in zip(
                                    reduced_items[i].quantity_state,
                                    reduced_items[j].quantity_state,
                                )
                            ),
                        }
                    )
                    reduced_items[j] = reduced_items[j].model_copy(
                        update={"quantity_state": tuple()}
                    )

        return self.__class__(
            items=tuple(
                item for item in reduced_items if item.quantity_state != tuple()
            )
        )

    def _to_index(self) -> int:
        """convert current quantity state to lexicographic index.
        Returns:
            int: lexicographic index of the current quantity state.
        """
        return convert_prod_combination_to_index(
            n=[item.required + item.group_size for item in self],
            combination=[
                [
                    sum(item.quantity_state[: i + 1]) + i
                    for i, _ in enumerate(item.quantity_state)
                ][:-1]
                for item in self
            ],
        )

    def _from_index(self, index: int) -> Self:
        """create ItemTable from lexicographic index.

        Args:
            index (int): lexicographic index.

        Returns:
            Self: generated ItemTable.
        """
        combination = convert_index_to_prod_combination(
            n=[item.required + item.group_size for item in self],
            k=[item.required for item in self],
            index=index,
        )

        items: list[ItemModel] = []

        for i, item in enumerate(self):
            group_comb = [-1] + list(combination[i]) + [item.required + item.group_size]
            items.append(
                item.model_copy(
                    update={
                        "quantity_state": tuple(
                            group_comb[i] - group_comb[i - 1] - 1
                            for i, _ in enumerate(group_comb[1:], start=1)
                        )
                    }
                )
            )
        return self.model_copy(update={"items": tuple(items)})

    def _prob_to_move(self, src_index: int, dst_index: int) -> float:
        """calculate probability to move from src state to dst state.

        Args:
            src_index (int): the index of the source state.
            dst_index (int): the index of the destination state.

        Returns:
            float: probability of moving from src to dst.
        """
        src = self._from_index(src_index)
        dst = self._from_index(dst_index)

        filtered = [
            {"src": src_item, "dst": dst_item}
            for src_item, dst_item in zip(src, dst)
            if src_item.quantity_state != dst_item.quantity_state
        ]

        # impossible process
        if len(filtered) > 1:
            return 0

        if len(filtered) == 0:
            return 1 - float(
                sum(item.prob * sum(item.quantity_state[:-1]) for item in src)
            )

        diff = filtered[0]
        for i, _ in enumerate(diff["src"].quantity_state[:-1]):
            if (
                diff["src"].quantity_state[i] - 1 == diff["dst"].quantity_state[i]
                and diff["src"].quantity_state[i + 1]
                == diff["dst"].quantity_state[i + 1] - 1
            ):
                return float(diff["src"].prob * diff["src"].quantity_state[i])
            elif diff["src"].quantity_state[i] != diff["dst"].quantity_state[i]:
                return 0

        raise AssertionError("Expected code to be unreachable.")

    def _generate_prob_matrix(self) -> np.ndarray:
        """generate probability matrix of the markov process.

        Returns:
            np.ndarray: probability matrix.
        """
        return np.array(
            [
                [self._prob_to_move(i, j) for j in list(range(self.matrix_size))[::-1]]
                for i in list(range(self.matrix_size))[::-1]
            ]
        )

    @timeout(5, use_signals=True)
    def set_caches(self) -> Self:
        """set caches for probability matrix and its inverse."""

        cache_mat = self._generate_prob_matrix()

        E = np.identity(self.matrix_size - 1)
        Q = cache_mat[:-1, :-1]
        cache_mat_inv = np.linalg.inv(E - Q)

        return self.model_copy(
            update={
                "cache_mat": cache_mat,
                "cache_mat_inv": cache_mat_inv,
            }
        )

    def _calc_average(self) -> float:
        """calculate average count to collect all items.

        Returns:
            float: average count.
        """
        PI = np.array([[1] + [0] * (self.matrix_size - 2)])
        ONE = np.array([[1] * (self.matrix_size - 1)]).T
        return (PI @ self.cache_mat_inv @ ONE)[0, 0]

    def _calc_std(self, cache_ave: float) -> float:
        """calculate standard deviation of count to collect all items.

        Returns:
            float: standard deviation.
        """
        PI = np.array([[1] + [0] * (self.matrix_size - 2)])
        E = np.identity(self.matrix_size - 1)
        Q = self.cache_mat[:-1, :-1]
        ONE = np.array([[1] * (self.matrix_size - 1)]).T
        return math.sqrt(
            (PI @ (E + Q) @ np.linalg.matrix_power(self.cache_mat_inv, 2) @ ONE)[0, 0]
            - cache_ave**2
        )

    @timeout(10, use_signals=True)
    def calc_pdf(self) -> list[float]:
        """calculate probability distribution function (pdf).

        Returns:
            list[float]: pdf values.
        """
        PI = np.array([[1] + [0] * (self.matrix_size - 2)])
        E = np.identity(self.matrix_size - 1)
        Q = self.cache_mat[:-1, :-1]
        ONE = np.array([[1] * (self.matrix_size - 1)]).T

        cdf: list[float] = [0]

        powered_q = np.identity(self.matrix_size - 1)
        while True:
            prob = (PI @ (E - powered_q) @ ONE)[0, 0]
            cdf.append(float(prob))
            if prob > 0.999:
                return [cdf[i + 1] - cdf[i] for i, _ in enumerate(cdf[:-1])]
            powered_q = powered_q @ Q

    def describe(self, pdf: list[float] | None = None) -> dict[str, float]:
        """get monte carlo simulation statistics.

        Args:
            pdf (list[float]): probability distribution function.

        Returns:
            tuple[float, float]: average and standard deviation.
        """
        properties: dict[str, float] = {}
        if pdf is not None:
            cdf = [sum(pdf[: i + 1]) for i, _ in enumerate(pdf)]
            for key in [
                "1%",
                "5%",
                "10%",
                "20%",
                "25%",
                "50%",
                "75%",
                "80%",
                "90%",
                "95%",
                "99%",
            ]:
                properties[key] = cdf.index(
                    next(i for i in cdf if i >= float(key.strip("%")) / 100)
                )

        properties["平均"] = self._calc_average()
        properties["標準偏差"] = self._calc_std(properties["平均"])

        if pdf is not None:
            properties["最頻値"] = pdf.index(max(pdf))

        return properties
