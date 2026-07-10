# Canonicalization wiring sweep (#389) — U1↔U2↔U3 producer/consumer map + wire-in

**Date:** 2026-07-09 · **Operator GO:** #389 ("comprehensive sweep to identify upstream and
downstream producers and consumers and ensure all wired and integrated and working as expected").
$0, no GPU, run dirs read-only. **Pointer contest-CPU 0.19110 UNMOVED** — this is MEANS (measurement
+ coordination apparatus), not a lever, not an exact-eval row. `[macOS-CPU advisory · NON-PROMOTABLE]`.

The 3 canonical units landed with `# TODO(#389)` fences; this sweep resolves the fences, wires the
cross-unit consumption, migrates the one genuine duplicated live callsite, and runs round-1 review.

---

## PHASE 1 — producer / consumer map (executed, not asserted)

Greps run over `src/tac` (see the sweep for the exact queries): the discriminating signal for the
through-R d_seg pattern is the `d_seg_reference` + `per_class_flip_stats` compare-consumers, NOT the
too-broad `preprocess_input` (every renderer defines that method).

| Canonical unit | Upstream producers (feed it) | Downstream consumers (read it) | #389 action |
|---|---|---|---|
| **U1 `tac.through_r`** (`measure_through_r`, `compare`, `scaffold_assembler`) | `resolution_chain` (pinned R) · `boundary_math.seg_core.load_real_segnet` · `witness_control.perclass_verdict` · gt_n600 cache | `inc1a_harness.mask_dseg_meter` (**migrated**) · `inc1a_harness.composite_assembler` (already re-exports scaffold) · `experiments/probe_*` (read-only evidence, not migrated) · **U2** (rows) | emit `MeasurementRow`s (fence resolved) + share compare back-half |
| **U2 `tac.verdicts`** (`MeasurementRow`, `emit_verdict`) | **U1** `ThroughRResult.to_measurement_rows` (new) · any caller building a verdict | `session_bus` (verdict fan-in) · `triality_drift_detector` (serializer `--triality-legs`) · dashboards/costate (duty-to-measure) | posts `verdict_landed` bulletin (fence resolved) |
| **U3 `tac.session_bus`** (`bulletin`, `recovery_manifest`) | `review_counter.record_round` (WIRED pre-#389) · **U2** `emit_verdict` (new) · **U3** `recovery_manifest.register_inflight/complete` (new) | seal-round `staleness_check` · `#247` costate SENSE `events_since` · `session_recover.py report` | add 2 fail-open producers |

**Honest Phase-1 findings (verified by reading the code, not the prompt's suggestion — operating
manual §4):**
- **`movable_deshare.py` is NOT a through-R consumer.** It is a *rate-axis* byte dedup audit (numpy
  geometry on the frozen argmax cache → `encode_absolute_2d`); it never renders through R, never runs
  SegNet, never calls `d_seg_reference`. Migrating it to `measure_through_r` would be a FAKE migration.
  Its `measure_deshare_magnitude` / `pairwise_dedup_audit` byte/S outputs are candidate **U2**
  `MeasurementRow` (rate-axis) emitters — noted as a future wire, not forced in this sweep.
- **`segmap_renderer.py`** `argmax(dim=1)` is a renderer-internal training compare, not the
  gt_n600→R→SegNet verdict pattern. Out of scope.
- **`scaffold_assembler` / `laguerre_logit_offset`** use `d_seg_reference` / `per_class_flip_stats`
  for their OWN single-frame reconciliation / task-space lever — a *different* computation than the
  agg+per-class stack-compare. Not the duplicated fact. Out of scope.
- **The one genuine duplication (P1):** the "compare a realized/partition INT label stack vs `lstars`
  → aggregate d_seg + per-class rate + flip-share" **back half** was computed twice, inline, by
  `measure_through_r` and `measure_mask_dseg`. That is the fact now stored in ONE place.

---

## PHASE 2 — wirings landed

1. **U1→U2 (fence resolved).** `ThroughRResult.to_measurement_rows(*, git_sha, review_status, …)`
   builds canonical `MeasurementRow`s (1 aggregate `d_seg` + 5 per-class), `axis_tag=[through-R]`
   (non-authority by construction), `n_samples` = frame count (subset carries its reason),
   `provenance.inputs_sha256` = a new deterministic `sha256(lstars ‖ realized)` key computed in
   `measure_through_r`. OPT-IN (a method, not the hot loop) + LAZY import ⇒ `measure_through_r` stays
   **byte-identical by default** and the harness stays leaf-clean at import time. `review_status` is a
   REQUIRED arg (the harness cannot know if a fresh reviewer saw the number — operating manual §5).
   *Deviation from the fence's "per pair" wording:* aggregate+per-class is the verdict-useful
   granularity; per-pair scalars remain in `per_pair_dseg` for a caller who wants 600 rows.
2. **U2→U3 (fence resolved).** `emit_verdict` posts a fail-open `verdict_landed` bulletin AFTER the
   atomic JSON write (`subject = scope.scoped_to`), mirroring `review_counter`. Lazy import ⇒
   import-time deps stay stdlib+`tac.verdicts`; the JSON write is unchanged/byte-identical.
3. **Migration (the one genuine callsite).** New canonical `tac.through_r.compare.
   compare_label_stack_to_lstars` holds the shared back half; **both** `measure_through_r` and
   `measure_mask_dseg` delegate to it (each keeps its own front-half validation + error type +
   `extra` key name). `mask_dseg_meter` "delegates into `through_r`, keeps API" — the honest reading
   of the prompt (mask-level ≠ through-R front half; only the back half is shared).
4. **U3 producers.** `recovery_manifest.register_inflight → agent_spawned`, `complete →
   agent_completed` (fail-open, AFTER the durable checkpoint write). The heartbeat path is
   deliberately NOT wired (would spam `agent_spawned` each tick). **memo-landing helper:** N/A — no
   `tac` function lands a memo (memos land via git); the low-level `subagent_checkpoint.append_checkpoint`
   is deliberately not wired (heartbeat spam) — the semantic register/complete layer is
   `recovery_manifest`.
5. All commits via the serializer with `--triality-legs` (dogfooding U2's flag).

### No-regression proofs (numeric identity — re-derived, not recognised)
- **`compare` == the old inline formula, EXACTLY** (`test_compare_helper_matches_inline_formula_exactly`):
  independent inline recompute of agg/per-class/share/flips/pixels/std `==` the helper, on a synthetic
  5-class stack with genuine flips.
- **`mask_dseg_meter` migration is numeric-identity** (`test_mask_meter_migration_is_numeric_identity`):
  same inputs → identical `agg_dseg` / `per_class_dseg` / `flip_share` / `total_flips` / `total_pixels`
  / `extra["per_frame_std"]` before/after (the `extra` key preserved = API unchanged).
- **U1 identity canary still GREEN post-wiring** (`test_end_to_end_gt_frame_reproduces_lstars_dseg_zero`
  RAN, not skipped — real gt cache + real SegNet): GT frame → `measure_through_r` (through the new
  delegated back half) → `agg d_seg == 0.0`, `total_flips == 0`. The strongest possible identity proof.

---

## PHASE 3 — round-1 adversarial review (I own round 1 over U1+U2+U3+the wirings)

Attacks and dispositions:
- **(a) serializer `--triality-legs` under real use** → dogfooded on this sweep's commits; 58
  serializer + 21 triality-legs tests green; malformed-flag refuses before any git action (existing).
- **(b) numeric identity of migrated callsites** → proven exact (synthetic ==) AND through the real
  SegNet (canary d_seg==0). HELD.
- **(c) fail-open handlers swallowing real errors** → each new fail-open post is AFTER the durable
  write on a score-neutral notification path (operating manual §8.9), documented at the callsite;
  tested that a raising `post_event` never breaks the verdict/checkpoint write. ACCEPTED risk (same
  precedent as `review_counter`).
- **(d) U1 identity canary green post-wiring** → confirmed RAN + passed. HELD.
- **(e) P1 "any fact now in TWO places?"** → the compare back half is now ONE place (`compare.py`); no
  NEW duplication introduced (`inputs_sha256` only in harness; each fail-open helper in one module).

**FINDING (found + fixed — round 1 is therefore NOT_CLEAN, counter stays 0):** wiring the fail-open
bulletin producers into `emit_verdict` and `recovery_manifest.register/complete` made the EXISTING
unit tests write real events into the live LIVE_STATE store `.omx/state/session_events.jsonl` (a
concurrent seal agent's `staleness_check` reads it). Measured leak: `test_verdict_emit` +7,
`test_session_bus` +4, `test_review_counter` +30. **Class-fix:** bulletin-store isolation added to
every emitting test surface — autouse fixture in `test_verdict_emit.py` + `test_review_counter.py`,
and `tool_tmp` + the CLI test extended in `test_session_bus.py`. Post-fix in-process leak = **0**.
**Residual (documented, out of this sweep's scope):** `test_review_counter`'s multiprocessing
concurrency test still leaks **12** events from subprocess children (spawned children don't inherit
`monkeypatch`; U3's `record_round` has no `bulletin_path` param to thread) — a deeper U3 hardening
item (add a bulletin-path param / env override), NOT a new defect in the #389 wiring. Flagged for
rounds 2+.

Round-1 verdict: **NO-GO-to-SEAL** (found + fixed 1 class) → recorded `NOT_CLEAN`, findings=1. Main
dispatches rounds 2+ fresh; SEAL needs 3 consecutive CLEAN.

---

## Tests / hygiene
- Full affected suite **264 passed** (through_r 32 · inc1a 27 · verdict_measurement_row 24 ·
  verdict_emit 17 · session_bus 34 · review_counter 14 · subagent_checkpoint 54 · serializer 58 ·
  serializer_triality 21 + new: compare_helper 7 · canon389_wiring 6). New: **13** tests.
- `ruff check` clean on every touched file (`flip_inverse.py`'s pre-existing debt is an untracked
  sibling file, not touched here).
- review-tracker mark-file per `.py`; serializer commit with POST-EDIT shas; `--triality-legs` used.

## Triality legs
- **DAG:** `FEED-389` in `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.
- **DSL:** N/A-with-reason — apparatus wiring (measurement/verdict/coordination), no witness lever /
  trainer flag / curriculum surface for `witness_dsl` to hold.
- **equations:** N/A-with-reason — reuses the already-registered `d_seg` authority functional + score
  law; the producer/consumer wiring is the invariant, no NEW measured physics relation.

STORES CONSULTED: the 3 canon landing memos (`canonical_{through_r_harness,verdict_emission,
session_bus}_landed_20260709.md`), `docs/operating_manual_craft_handoff.md`, the DAG (FEED-canon-u1/u2/u3),
`tac.review_counter` (fail-open producer template), `tac.through_r.*`, `tac.verdicts.*`,
`tac.session_bus.*`, `tac.inc1a_harness.mask_dseg_meter`, `tac.boundary_math.movable_deshare`.
