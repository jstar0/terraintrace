"""Synthetic grouped terrain fixtures with no external raster dependency."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import TerrainCase, validate_scale


@dataclass(frozen=True, slots=True)
class TerrainDataset:
    """Train/validation/test cases split by synthetic region group."""

    train: tuple[TerrainCase, ...]
    validation: tuple[TerrainCase, ...]
    test: tuple[TerrainCase, ...]

    def __post_init__(self) -> None:
        splits = (self.train, self.validation, self.test)
        if any(not split for split in splits):
            raise ValueError("each dataset split must contain at least one case")
        ids = [case.id for split in splits for case in split]
        split_groups = [set(case.group for case in split) for split in splits]
        if len(ids) != len(set(ids)):
            raise ValueError("case ids must be unique across splits")
        if any(left & right for index, left in enumerate(split_groups) for right in split_groups[index + 1 :]):
            raise ValueError("terrain groups must be disjoint across splits")

    @property
    def scale(self) -> int:
        scales = {case.scale for split in (self.train, self.validation, self.test) for case in split}
        if len(scales) != 1:
            raise ValueError("all cases in a dataset must use one scale")
        return scales.pop()

    def all_cases(self) -> tuple[TerrainCase, ...]:
        return self.train + self.validation + self.test


def create_dataset(seed: int = 17, *, size: int = 48, scale: int = 4) -> TerrainDataset:
    """Create a deterministic 12-case terrain set with group-held-out splits."""

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    validate_scale(scale)
    if size < 16 or size % scale:
        raise ValueError("size must be at least 16 and divisible by scale")
    rng = np.random.default_rng(seed)
    groups = ("ridge-basin", "coastal-shelf", "valley-plateau")
    cases: list[TerrainCase] = []
    for group_index, group in enumerate(groups):
        for case_index in range(4):
            case_seed = int(rng.integers(0, 2**32 - 1))
            case_rng = np.random.default_rng(case_seed)
            values = _terrain_surface(case_rng, size, group_index, case_index)
            cases.append(
                TerrainCase(
                    id=f"{group_index}-{case_index:02d}",
                    group=group,
                    high_res=values,
                    scale=scale,
                    pixel_size=1.0,
                )
            )
    return TerrainDataset(
        train=tuple(cases[:4]),
        validation=tuple(cases[4:8]),
        test=tuple(cases[8:12]),
    )


def _terrain_surface(rng: np.random.Generator, size: int, group_index: int, case_index: int) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, size, dtype=np.float64)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    phase = 0.35 * case_index + 0.2 * group_index
    slope = (2.2 + 0.4 * group_index) * x + (1.1 - 0.15 * case_index) * y
    ridge = (5.0 + group_index) * np.exp(-((y - 0.35 * np.sin(2.4 * x + phase)) ** 2) / 0.035)
    basin = -(3.5 + 0.4 * case_index) * np.exp(-(((x + 0.25 * np.cos(phase)) ** 2) + (y - 0.2) ** 2) / 0.18)
    waves = (0.8 + 0.1 * group_index) * np.sin(4.0 * x + phase) * np.cos(3.0 * y - phase)
    plateau = np.where(x + 0.45 * y > 0.25 + 0.05 * case_index, 1.4, -0.2)
    micro = rng.normal(0.0, 0.055 + 0.01 * group_index, size=(size, size))
    return 100.0 + slope + ridge + basin + waves + plateau + micro
