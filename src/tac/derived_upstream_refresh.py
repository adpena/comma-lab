# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "tac.derived_upstream_refresh_registry.v1"

BASE_SHA_PRESENT = "PRESENT"
BASE_SHA_UNKNOWN = "UNKNOWN_IN_UF1_SCOPE"
BASE_SHA_FORMULA_INVARIANT = "FORMULA_INVARIANT"

DISPOSITION_CURRENT = "CURRENT"
DISPOSITION_EXACT_INVARIANT = "EXACT_INVARIANT"
DISPOSITION_REFRESHED_SCORER_FREE = "REFRESHED_SCORER_FREE"
DISPOSITION_QUEUED_HEAVY_REFRESH = "QUEUED_HEAVY_REFRESH"
DISPOSITION_QUEUED_FIBER_INPUT_BLOCKED = "QUEUED_FIBER_INPUT_BLOCKED"

ROUTE_ALREADY_CURRENT = "already_current"
ROUTE_EXACT_INVARIANT = "exact_invariant"
ROUTE_FIBER_TRANSPORT = "fiber_transport"
ROUTE_FULL_RECOMPUTE = "full_recompute"
ROUTE_SCORER_FREE_DERIVATION = "scorer_free_derivation"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

VALID_BASE_SHA_STATUSES = frozenset(
    {
        BASE_SHA_PRESENT,
        BASE_SHA_UNKNOWN,
        BASE_SHA_FORMULA_INVARIANT,
    }
)
VALID_DISPOSITIONS = frozenset(
    {
        DISPOSITION_CURRENT,
        DISPOSITION_EXACT_INVARIANT,
        DISPOSITION_REFRESHED_SCORER_FREE,
        DISPOSITION_QUEUED_HEAVY_REFRESH,
        DISPOSITION_QUEUED_FIBER_INPUT_BLOCKED,
    }
)
VALID_REFRESH_ROUTES = frozenset(
    {
        ROUTE_ALREADY_CURRENT,
        ROUTE_EXACT_INVARIANT,
        ROUTE_FIBER_TRANSPORT,
        ROUTE_FULL_RECOMPUTE,
        ROUTE_SCORER_FREE_DERIVATION,
    }
)
VALID_RADIUS_STATUSES = frozenset({"KNOWN", "UNKNOWN", "NOT_APPLICABLE"})


class RefreshRegistryError(ValueError):
    """Raised when a derived upstream quantity is stale or malformed."""


def _require_nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RefreshRegistryError(f"{name} must be a non-empty string")
    return value


def _require_sha256_or_none(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RefreshRegistryError(f"{name} must be a lowercase sha256, got {value!r}")
    return value


def _string_tuple(values: Iterable[str], name: str) -> tuple[str, ...]:
    out = tuple(values)
    if not out:
        raise RefreshRegistryError(f"{name} must not be empty")
    for item in out:
        _require_nonempty(item, name)
    return out


@dataclass(frozen=True)
class RefreshRegistryRow:
    """One derived quantity plus its freshness contract at the consuming surface."""

    quantity_id: str
    description: str
    base_identity_kind: str
    computed_at_base_sha256: str | None
    base_sha_status: str
    base_age: str
    current_base_sha256: str | None
    consumers: tuple[str, ...]
    validity_radius_status: str
    validity_radius_derive_route: str
    refresh_route: str
    trigger: str
    owner: str
    disposition: str
    evidence_paths: tuple[str, ...] = field(default_factory=tuple)
    transported_to_base_sha256: str | None = None
    score_claim: bool = False
    promotion_eligible: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        _require_nonempty(self.quantity_id, "quantity_id")
        _require_nonempty(self.description, "description")
        _require_nonempty(self.base_identity_kind, "base_identity_kind")
        _require_nonempty(self.base_age, "base_age")
        _require_nonempty(self.validity_radius_derive_route, "validity_radius_derive_route")
        _require_nonempty(self.trigger, "trigger")
        _require_nonempty(self.owner, "owner")
        _string_tuple(self.consumers, "consumers")
        for path in self.evidence_paths:
            _require_nonempty(path, "evidence_paths")
        if self.base_sha_status not in VALID_BASE_SHA_STATUSES:
            raise RefreshRegistryError(f"unknown base_sha_status {self.base_sha_status!r}")
        if self.disposition not in VALID_DISPOSITIONS:
            raise RefreshRegistryError(f"unknown disposition {self.disposition!r}")
        if self.refresh_route not in VALID_REFRESH_ROUTES:
            raise RefreshRegistryError(f"unknown refresh_route {self.refresh_route!r}")
        if self.validity_radius_status not in VALID_RADIUS_STATUSES:
            raise RefreshRegistryError(
                f"unknown validity_radius_status {self.validity_radius_status!r}"
            )
        _require_sha256_or_none(self.computed_at_base_sha256, "computed_at_base_sha256")
        _require_sha256_or_none(self.current_base_sha256, "current_base_sha256")
        _require_sha256_or_none(self.transported_to_base_sha256, "transported_to_base_sha256")
        if self.base_sha_status == BASE_SHA_PRESENT and self.computed_at_base_sha256 is None:
            raise RefreshRegistryError(
                f"{self.quantity_id}: PRESENT base sha status requires computed_at_base_sha256"
            )
        if self.base_sha_status == BASE_SHA_UNKNOWN and self.computed_at_base_sha256 is not None:
            raise RefreshRegistryError(
                f"{self.quantity_id}: UNKNOWN base sha status must not carry a fake sha"
            )
        if self.base_sha_status == BASE_SHA_FORMULA_INVARIANT:
            if self.refresh_route != ROUTE_EXACT_INVARIANT:
                raise RefreshRegistryError(
                    f"{self.quantity_id}: formula invariant rows must use exact_invariant route"
                )
            if self.disposition != DISPOSITION_EXACT_INVARIANT:
                raise RefreshRegistryError(
                    f"{self.quantity_id}: formula invariant rows must have EXACT_INVARIANT "
                    "disposition"
                )

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "quantity_id": self.quantity_id,
            "description": self.description,
            "base_identity_kind": self.base_identity_kind,
            "computed_at_base_sha256": self.computed_at_base_sha256,
            "base_sha_status": self.base_sha_status,
            "base_age": self.base_age,
            "current_base_sha256": self.current_base_sha256,
            "consumers": list(self.consumers),
            "validity_radius": {
                "status": self.validity_radius_status,
                "derive_route": self.validity_radius_derive_route,
            },
            "refresh_route": self.refresh_route,
            "trigger": self.trigger,
            "owner": self.owner,
            "disposition": self.disposition,
            "evidence_paths": list(self.evidence_paths),
            "transported_to_base_sha256": self.transported_to_base_sha256,
            "score_claim": self.score_claim,
            "promotion_eligible": self.promotion_eligible,
            "notes": self.notes,
        }

    @classmethod
    def from_json_obj(cls, obj: Mapping[str, Any]) -> RefreshRegistryRow:
        if obj.get("schema") != SCHEMA:
            raise RefreshRegistryError(f"unknown schema {obj.get('schema')!r}")
        validity = obj.get("validity_radius")
        if not isinstance(validity, Mapping):
            raise RefreshRegistryError(f"{obj.get('quantity_id')}: validity_radius is required")
        return cls(
            quantity_id=str(obj.get("quantity_id", "")),
            description=str(obj.get("description", "")),
            base_identity_kind=str(obj.get("base_identity_kind", "")),
            computed_at_base_sha256=obj.get("computed_at_base_sha256"),
            base_sha_status=str(obj.get("base_sha_status", "")),
            base_age=str(obj.get("base_age", "")),
            current_base_sha256=obj.get("current_base_sha256"),
            consumers=tuple(obj.get("consumers", ())),
            validity_radius_status=str(validity.get("status", "")),
            validity_radius_derive_route=str(validity.get("derive_route", "")),
            refresh_route=str(obj.get("refresh_route", "")),
            trigger=str(obj.get("trigger", "")),
            owner=str(obj.get("owner", "")),
            disposition=str(obj.get("disposition", "")),
            evidence_paths=tuple(obj.get("evidence_paths", ())),
            transported_to_base_sha256=obj.get("transported_to_base_sha256"),
            score_claim=bool(obj.get("score_claim", False)),
            promotion_eligible=bool(obj.get("promotion_eligible", False)),
            notes=str(obj.get("notes", "")),
        )


def require_fresh_for_consumption(
    row: RefreshRegistryRow,
    *,
    current_base_sha256: str,
    consumer: str,
) -> None:
    """Fail closed before a consumer uses a derived stale-able quantity."""
    _require_sha256_or_none(current_base_sha256, "current_base_sha256")
    if consumer not in row.consumers:
        raise RefreshRegistryError(
            f"{row.quantity_id}: consumer {consumer!r} is not declared; declared consumers "
            f"are {row.consumers!r}"
        )
    if row.disposition == DISPOSITION_EXACT_INVARIANT:
        return
    if row.disposition in (DISPOSITION_QUEUED_HEAVY_REFRESH, DISPOSITION_QUEUED_FIBER_INPUT_BLOCKED):
        raise RefreshRegistryError(
            f"{row.quantity_id}: {row.disposition}; route={row.refresh_route}; "
            f"trigger={row.trigger}; owner={row.owner}"
        )
    if row.computed_at_base_sha256 == current_base_sha256:
        return
    if row.transported_to_base_sha256 == current_base_sha256:
        return
    raise RefreshRegistryError(
        f"{row.quantity_id}: stale for {consumer}; computed_at={row.computed_at_base_sha256}, "
        f"current={current_base_sha256}, route={row.refresh_route}, trigger={row.trigger}, "
        f"owner={row.owner}"
    )


def registry_denominators(rows: Iterable[RefreshRegistryRow]) -> dict[str, int]:
    materialized = tuple(rows)
    return {
        "quantities_found": len(materialized),
        "with_consumers": sum(1 for row in materialized if row.consumers),
        "with_triggers": sum(1 for row in materialized if row.trigger.strip()),
        "with_known_validity_radius": sum(
            1 for row in materialized if row.validity_radius_status == "KNOWN"
        ),
        "scorer_free_refreshed": sum(
            1 for row in materialized if row.disposition == DISPOSITION_REFRESHED_SCORER_FREE
        ),
        "exact_invariants": sum(
            1 for row in materialized if row.disposition == DISPOSITION_EXACT_INVARIANT
        ),
        "heavy_refreshes_queued": sum(
            1 for row in materialized if row.disposition == DISPOSITION_QUEUED_HEAVY_REFRESH
        ),
        "fiber_input_blockers_queued": sum(
            1
            for row in materialized
            if row.disposition == DISPOSITION_QUEUED_FIBER_INPUT_BLOCKED
        ),
        "current_rows": sum(1 for row in materialized if row.disposition == DISPOSITION_CURRENT),
    }


def load_refresh_registry_jsonl(path: str | Path) -> tuple[RefreshRegistryRow, ...]:
    rows: list[RefreshRegistryRow] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(RefreshRegistryRow.from_json_obj(json.loads(line)))
        except (json.JSONDecodeError, RefreshRegistryError) as exc:
            raise RefreshRegistryError(f"{path}:{lineno}: {exc}") from exc
    if not rows:
        raise RefreshRegistryError(f"{path}: empty registry is vacuous")
    return tuple(rows)


def write_refresh_registry_jsonl(rows: Iterable[RefreshRegistryRow], path: str | Path) -> None:
    materialized = tuple(rows)
    if not materialized:
        raise RefreshRegistryError("refusing to write an empty refresh registry")
    text = "".join(
        json.dumps(row.to_json_obj(), sort_keys=True, separators=(",", ":")) + "\n"
        for row in materialized
    )
    Path(path).write_text(text, encoding="utf-8")
