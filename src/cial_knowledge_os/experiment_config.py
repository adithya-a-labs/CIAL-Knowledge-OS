"""Declarative, pipeline-agnostic experiment sweep configuration."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


def _stable_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _stable_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_stable_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """One immutable configuration in an experiment sweep."""

    experiment_id: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"experiment_id": self.experiment_id, **dict(self.parameters)}


@dataclass(frozen=True, slots=True)
class ExperimentGrid:
    """Generate a deterministic Cartesian product from arbitrary parameters."""

    parameters: Mapping[str, Sequence[Any]]
    id_prefix: str = "experiment"

    def expand(self) -> list[ExperimentConfig]:
        if not self.parameters:
            return [ExperimentConfig(f"{self.id_prefix}_001", {})]
        names = list(self.parameters)
        values: list[Sequence[Any]] = []
        for name in names:
            options = self.parameters[name]
            if not options:
                raise ValueError(f"Experiment parameter '{name}' has no values.")
            values.append(options)
        return [
            ExperimentConfig(
                experiment_id=f"{self.id_prefix}_{index:03d}",
                parameters=dict(zip(names, combination, strict=True)),
            )
            for index, combination in enumerate(
                itertools.product(*values),
                start=1,
            )
        ]


def configuration_fingerprint(parameters: Mapping[str, Any]) -> str:
    """Return a short stable identifier useful across future pipeline phases."""

    payload = json.dumps(
        _stable_value(parameters),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def ensure_experiment_configs(
    values: ExperimentGrid | Iterable[ExperimentConfig | Mapping[str, Any]],
) -> list[ExperimentConfig]:
    """Normalize grids and mapping iterables to numbered configurations."""

    if isinstance(values, ExperimentGrid):
        return values.expand()
    configs: list[ExperimentConfig] = []
    for index, value in enumerate(values, start=1):
        if isinstance(value, ExperimentConfig):
            configs.append(value)
        else:
            parameters = dict(value)
            experiment_id = str(
                parameters.pop("experiment_id", f"experiment_{index:03d}")
            )
            configs.append(ExperimentConfig(experiment_id, parameters))
    if not configs:
        raise ValueError("At least one experiment configuration is required.")
    ids = [config.experiment_id for config in configs]
    if len(ids) != len(set(ids)):
        raise ValueError("Experiment IDs must be unique.")
    return configs
