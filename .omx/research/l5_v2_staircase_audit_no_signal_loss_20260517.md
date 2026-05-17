# L5 v2 Staircase Audit No-Signal-Loss Ledger - 2026-05-17

## Purpose

Preserve the completed L5 v2 staircase audit signal before the next main-branch
push. This ledger is not a score claim and does not promote or retire a lane.
It records implementation-hardening findings that should drive the next L5/L5
v2 staircase patch wave.

## Custody

- Repo: `/Users/adpena/Projects/pact`
- Branch: `main`
- Source of truth: `origin/main`
- Preserved from completed read-only audit agents:
  - `019e33ac-f69a-76d2-a575-b4a677684571`
  - `019e33e7-7a73-7130-b9b5-d308f0c977a5`
  - `019e33f0-35ab-7052-aeb1-275dd452a333`

## Durable Findings

1. TT5L/L5 v2 reusable surfaces already exist and should be reused instead of
   reimplemented:
   - `src/tac/substrates/time_traveler_l5_autonomy/archive.py`
   - `src/tac/substrates/time_traveler_l5_autonomy/consumption_proof.py`
   - `src/tac/optimization/l5_v2_sideinfo_effect_curve.py`
   - `src/tac/optimization/l5_v2_measurement_schedule.py`
   - `src/tac/optimization/l5_v2_paired_measurement_dispatch_plan.py`

2. Current L5 v2 side-info evidence must remain false-authority-resistant.
   A trained packet with all-zero side-info liveness is a negative anchor or
   blocker, not a staircase step. Variant manifests must make no-op status,
   mutated offsets, section hashes, and parser consumption explicit.

3. Effect-curve validation must require real artifact custody for promoted or
   schedule-driving cells. A hand-written curve with non-empty path strings is
   insufficient unless the validator checks those paths relative to the repo or
   declared artifact base and verifies exact-eval custody fields.

4. Side-info variant packets need no-op and byte-closure guards strong enough
   to reject unchanged archive SHA, unchanged member SHA, or unchanged side-info
   section SHA when a variant is expected to change scored bytes.

5. Human-readable packet reports should expose the same no-op evidence that the
   JSON manifests carry: generation rule, seed/source, archive/member hashes,
   side-section hashes, mutated offsets, and liveness summary.

6. Deterministic-reproducibility language must match implementation reality. If
   generated manifests include wall-clock timestamps such as `generated_at_utc`,
   either the docs must stop promising byte-identical deterministic JSON, or the
   timestamp must move outside the deterministic payload.

## Next Patch Targets

1. Re-inspect `validate_l5_v2_sideinfo_effect_curve()` and
   `_sideinfo_effect_curve_blockers()` for artifact-base handling. If any
   public path still defaults to CWD for custody validation, make the base
   explicit and add a negative test.

2. Re-inspect `tt5l_sideinfo_variant_packets.py` for unchanged-archive and
   unchanged-side-section blockers, then add any missing regression tests.

3. Extend the L5 v2 markdown/report surface so JSON-only no-op evidence is
   visible during operator review.

4. Keep all future L5/L5 v2 staircase work on `main`, with exact CPU/CUDA axis
   labels preserved and no score promotion from proxy or custody-light signals.

## Verification Intent

The next code patch should run the focused L5 suite at minimum:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  src/tac/tests/test_l5_staircase_v2.py \
  src/tac/tests/test_l5_v2_measurement_schedule.py \
  src/tac/tests/test_tt5l_sideinfo_variant_packets.py \
  -p no:cacheprovider
```

This ledger intentionally captures the audit signal before any implementation
patch so there is no loss if the next work wave is interrupted.
