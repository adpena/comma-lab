"""Scorer metadata and registry helpers for task-aware codec experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class ScorerSpec:
    """Serializable scorer description.

    The callable itself is kept outside this record so metadata can be stored,
    compared, or serialized independently from runtime execution.
    """

    name: str
    family: str | None = None
    description: str | None = None
    outputs: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RegisteredScorer:
    """Runtime scorer wrapper that keeps metadata attached to a callable."""

    spec: ScorerSpec
    scorer: Callable[..., Any] = field(repr=False, compare=False)

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def family(self) -> str | None:
        return self.spec.family

    @property
    def description(self) -> str | None:
        return self.spec.description

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.spec.outputs

    @property
    def metadata(self) -> Mapping[str, object]:
        return self.spec.metadata

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.scorer(*args, **kwargs)


class ScorerRegistry:
    """Small in-memory registry for task-aware scorer callables."""

    def __init__(self) -> None:
        self._scorers: dict[str, RegisteredScorer] = {}

    def register(
        self,
        name: str,
        scorer: Callable[..., Any],
        *,
        family: str | None = None,
        description: str | None = None,
        outputs: tuple[str, ...] | list[str] = (),
        **metadata: object,
    ) -> RegisteredScorer:
        if name in self._scorers:
            raise ValueError(f"Scorer already registered: {name}")
        spec = ScorerSpec(
            name=name,
            family=family,
            description=description,
            outputs=tuple(outputs),
            metadata=dict(metadata),
        )
        registered = RegisteredScorer(spec=spec, scorer=scorer)
        self._scorers[name] = registered
        return registered

    def get(self, name: str) -> RegisteredScorer:
        try:
            return self._scorers[name]
        except KeyError as exc:
            raise KeyError(f"Unknown scorer: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._scorers))

    def items(self) -> tuple[RegisteredScorer, ...]:
        return tuple(self._scorers[name] for name in self.names())
