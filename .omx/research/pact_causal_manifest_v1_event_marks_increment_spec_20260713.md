# `pact.causal_manifest.v1` marked-event telemetry increment — D39 specification

**Date:** 2026-07-13
**Status:** SPEC-ONLY / implementation ticketed / `src/tac/causal_manifest.py` unchanged
**Authority:** observability-only, `NON-PROMOTABLE`
**Schema:** additive `row_kind="event_mark"` under existing `pact.causal_manifest.v1`

## 1. Requirement and non-requirement

Rung 4 requires **marks, not counts**. A binary `event=true` or an integer event
count cannot close

\[
H(E_t\mid X_t,C_t)
\]

because the decoder still needs the event family, active class edge, spacetime
address, and attachment/incidence change. The increment therefore records one
immutable row per marked prediction break. It remains score-neutral telemetry:
it neither asserts Markov sufficiency nor carries a score or promotion verdict.

The disjoint family assignment is the rung-4 priority partition:

\[
E_{pred}=E_{top}\;\dot\cup\;
(E_{chart}\setminus E_{top})\;\dot\cup\;
(E_R\setminus(E_{top}\cup E_{chart})).
\]

An event satisfying more than one detector is recorded exactly once at the
highest-priority family, while all matched detectors remain in evidence.

## 2. Additive row kind

Proposed module constant:

```python
ROW_EVENT_MARK = "event_mark"
```

`ROW_EVENT_MARK` is added to `_ROW_KINDS` in the same landing as its parser,
dataclasses, writer, conflict rules, and tests. Existing v1 rows retain their
bytes and semantics. Old readers that reject unknown row kinds are not silently
treated as forward-compatible; producer and strict reader update atomically.

## 3. Typed row

```text
EventMarkRow
  schema_id: Literal["pact.causal_manifest.v1"]
  row_kind: Literal["event_mark"]
  event_id: nonempty string
  run_id: nonempty string
  stage_id: nonempty string
  checkpoint_id: string | null
  pair_index: int >= 0
  frame_from: int >= 0
  frame_to: int > frame_from
  observed_at_utc: canonical UTC string
  authority_axis: Literal["[observability-only] NON-PROMOTABLE"]
  family: Literal["topology", "chart", "receiver_lattice"]
  kind: family-specific enum below
  detectors_matched: sorted nonempty tuple[str, ...]
  class_edge: ClassEdgeMark
  location: SpacetimeMark
  attachment: IncidenceMark
  stratum_before: StratumMark
  stratum_after: StratumMark
  receiver_state: ReceiverStateMark
  evidence: sorted tuple[ArtifactRef, ...]
  receiver_derivable: bool
  public_derivation_ref: ArtifactRef | null
  resume_key: nonempty string
  notes: sorted tuple[JsonField, ...]
```

### `ClassEdgeMark`

```text
winner_class: int in [0,4]
other_class: int in [0,4], other_class != winner_class
directed: bool
junction_classes: sorted tuple[int, ...]  # empty for an ordinary edge
```

The row never stores only `class_id`; the active directed edge or junction
class set is part of the event mark.

### `SpacetimeMark`

```text
coordinate_system: Literal["scorer_grid", "camera_grid", "latent_chart"]
y: finite float
x: finite float
time_fraction: finite float in [0,1]
support_y0, support_x0, support_y1, support_x1: finite float
chart_id: nonempty string
location_quantizer_id: nonempty string
```

The support box permits a merge/split/birth region; `(x,y,time_fraction)` is its
canonical representative. The quantizer id prevents a floating-point address
from acquiring unstated byte authority.

### `IncidenceMark`

```text
before_component_ids: sorted tuple[str, ...]
after_component_ids: sorted tuple[str, ...]
before_junction_ids: sorted tuple[str, ...]
after_junction_ids: sorted tuple[str, ...]
parent_child_edges: sorted tuple[(str, str), ...]
incidence_before_sha256: DigestRef | null
incidence_after_sha256: DigestRef | null
attachment_rule_id: nonempty string
```

At least one before/after component, junction, parent-child edge, or incidence
digest must be present. This is the strict guard that rejects count-only rows.

### `StratumMark`

The common refinement `sigma=(kappa,omega,a,r)` is recorded, without claiming
that the four coordinates are globally complete:

```text
topology_signature_id: nonempty string      # kappa
orbit_stabilizer_chart_id: nonempty string  # omega
activation_chart_id: nonempty string        # a
receiver_phase_cell_id: nonempty string     # r
```

### `ReceiverStateMark`

```text
R_operator_id: nonempty string
uint8_rounding_id: nonempty string
xi_quantizer_id: nonempty string
xi_symbol: sorted tuple[int, ...]
phase_symbol: int | null
prediction_chart_id: nonempty string
```

The `xi_symbol` is the actual quantized receiver datum, never an uncharged
continuous vector substituted for counted state.

## 4. Family and kind validation

```text
topology:
  component_birth | component_death | merge | split |
  hole_birth | hole_death | junction_incidence_change

chart:
  atlas_transition | stabilizer_change | admissible_arrow_change |
  occlusion | disocclusion | nonrigid_residual |
  clamp_cell_change | relu_cell_change | argmax_cell_change

receiver_lattice:
  resize_cell_crossing | uint8_rounding_crossing |
  subpixel_phase_wrap | sampled_connectivity_change
```

Validation enforces family/kind membership. `detectors_matched` may name lower
priority detectors, but the stored `family` must equal the first matched family
in priority order `topology > chart > receiver_lattice`.

## 5. Stable identity, append, and resume semantics

`event_id` is the hex SHA-256 of canonical JSON over:

```text
(run_id, stage_id, pair_index, frame_from, frame_to, family, kind,
 class_edge, location quantized under location_quantizer_id, attachment)
```

`resume_key` is the stage-local checkpoint cursor
`run_id/stage_id/pair_index/frame_to/event_id`. Append behavior follows the
manifest's immutable-identifier contract:

- missing id: append under the canonical lock;
- identical canonical row already present: idempotent no-op;
- same id with different bytes: `CausalManifestConflictError`;
- later evidence: a new row with a new id or an explicit future evidence-link
  row, never mutation of the old event.

A stage checkpoint records the last fully appended resume key. Restart scans to
that key and continues; at most one uncommitted event-detection interval is
recomputed.

## 6. Required implementation tests

1. strict round-trip for each family;
2. reject family/kind mismatch;
3. reject class-only or count-only marks;
4. reject absent attachment/incidence;
5. priority partition records an overlapping detector set once;
6. stable event id is independent of dict insertion order;
7. identical resume append is idempotent;
8. conflicting immutable id fails closed;
9. old v1 rows still parse byte-for-byte;
10. observability axis is fixed and score/promotion fields are rejected.

## 7. Consumer wire-in

- **Sensitivity map:** event surprise `-log2 q(event_mark|X,C)` may become a
  costate only after a calibrated model exists.
- **Pareto constraint:** telemetry cannot promote a witness or move a pointer.
- **Bit allocator:** price event family + mark payload separately from regular
  phase/residual symbols.
- **Cathedral/autopilot:** use event rows to select regular versus marked branch;
  fail closed if a nonzero event lacks its mark.
- **Continual learning:** aggregate only typed family/kind/calibration outcomes,
  never bare counts.
- **Probe disambiguator:** compare topology-only, full priority partition, and
  receiver-lattice-only detectors on the same immutable rows.

Implementation authority is the separate ticket
`.omx/research/TICKET_D39_event_marks_telemetry_20260713.md`.
