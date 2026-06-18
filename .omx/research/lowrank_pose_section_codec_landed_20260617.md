# Low-rank pose-section codec — LANDED (task #140, the REOPENED #1 finding)

**Date:** 2026-06-17
**Author:** torch-vehicle codec subagent (recursive design→build→wire→test→review)
**Authority:** torch-CPU TRUSTED. `[contest-CPU advisory]` NON-PROMOTABLE — no exact
`upstream/evaluate.py` row yet; the exact frontier pointer is UNMOVED.
**Scope:** $0, no GPU, no PR. Did NOT touch the live `launch_bind_all_taper_ab.py`
MPS run or its `experiments/results/bindall_arm_b_*` files.

## What landed

A low-rank SVD codec for the FiLM-STORE pose section, as an OPT-IN, DEFAULT-OFF
archive-grammar primitive in `tac.torch_vehicle.pose_film` + driver wire-in:

- `encode_pose_section_lowrank(stored_pose, *, rank=4, levels=511)` / `decode_pose_section_lowrank`
  — stores μ + the top-`rank` right-singular basis `Vt[:rank]` + the `rank` quantized,
  delta-zigzag-brotli'd principal time-series. New section magic `PFL2` (distinct from
  the legacy `PFLM`) → 100% unambiguous auto-dispatch.
- `lowrank_pose_section_fidelity(...) -> (bytes, mse)` — the Catalog #304 empirical
  bit-spend + fidelity proof (encode + round-trip MEASURE; nothing asserted).
- `decode_pose_section` / `parse_pose_section` now auto-dispatch on the 4-byte section
  magic, so a legacy `PFLM` archive and a new `PFL2` archive both decode through the
  SAME public decode + inflate path. Legacy decode is bit-equal to before.
- `build_archive_with_pose(..., pose_codec="iid"|"lowrank", lowrank_rank, lowrank_levels)`
  — DEFAULT `"iid"` is BYTE-IDENTICAL to today (proved by test).
- Driver config `pose_section_codec` (default `"iid"`) + `pose_section_lowrank_rank`
  (4) / `pose_section_lowrank_levels` (511), validated fail-closed in `__post_init__`,
  threaded into the FiLM byte-close path. `pose_film_v2` re-exports the new symbols
  (the live `--pose-film-v2` path emits a PFLM section, so the codec applies directly
  when opted in).

## Applicability gate (done FIRST)

The live `--pose-film-v2` run sets `pose_film_enabled=True` + `pose_film_version=2`,
so at byte-close `_build_archive_and_eval_decoder` takes the FiLM branch and emits a
real `PFLM` pose section via `encode_pose_section(stored_pose)`. The codec therefore
applies directly to the live pose-STORE path (opt-in), AND is a reusable primitive for
any pose-STORE archive. The default stays `iid` → the live run is byte-unaffected.

## The HONEST finding (the recursive-review correction)

The original reopened claim ("rank-2/254 = 2.70× smaller, lossless-relative-to-d_pose,
saves ~0.0013") does NOT survive the full-score adversarial math (Pass-1 finding):

- The byte/RATE saving is REAL + EXACT.
- BUT the legacy iid codec already stores the pose nearly losslessly (MSE ≈ 2.9e-5),
  and the contest pose term `sqrt(10·d_pose)` is highly nonlinear at this operating
  point (∂/∂d_pose ≈ 85). Trading fidelity (MSE 2.9e-5 → 2.6e-4 at rank-2/254) for
  bytes costs MORE on the pose term (upper-bound +0.017) than the bytes save on rate
  (−0.0013), IF storage-MSE maps ≥ ~1:1 to contest d_pose. So **rank-2/254 is
  net-NEGATIVE**.
- There IS a Pareto-DOMINANT point on the real pose: **rank-4/511 = 2563 B (smaller
  than iid's 3088 B) AND MSE 2.7e-5 (lower than iid's 2.9e-5)** → improves rate
  (−0.00035) while the pose term cannot worsen (MSE went down). That is the DEFAULT.
- The net win is a modest, unambiguous **~0.5 KB / −0.0004 rate**, advisory until a
  byte-closed `upstream/evaluate.py` (the storage-MSE → contest-d_pose mapping may be
  weaker than 1:1 since the decoder FiLM-CONDITIONS on the pose rather than
  reproducing it — a measured question, never asserted). This is much smaller than the
  reopened claim, but it is HONEST and Pareto-dominant (CLAUDE.md: a smaller honest
  result beats a larger fake one).

All findings are encoded as NO-FAKE tests so they cannot be re-asserted naively.

## Tests (23 NO-FAKE, all pass; behavior not constants)

round-trip fidelity on REAL pose · raw byte ratio · net-negative-at-naive-point
(recorded) · Pareto-dominant default exists · byte-identical default · legacy PFLM
decode unchanged · distinct magics · auto-dispatch (iid/lowrank/none/garbage) ·
rank+levels monotonicity · rank clamp · levels guard (incl. uint16-zigzag overflow
guard, Pass-4 fix) · magic guard · driver end-to-end build+inflate+magic · driver
default byte-identical · __post_init__ validation. Mutation-tested: a stub codec FAILS
the suite. 192 broader torch_vehicle pose/config/driver tests green (0 regressions).

## Recursive review log (3 clean passes after the last reset)

P1 FINDING (net-negative naive point → dominant default + honest framing) → reset ·
P2 clean · P3 clean (mutation) · P4 FINDING (uint16 overflow guard) → reset · P5 clean
· P6 FINDING (doc-drift) → reset · P7/P8/P9 clean (SEAL).

## 6-hook wire-in (CLAUDE.md Subagent coherence)

1. **sensitivity-map** — N/A (codec primitive; the per-byte pose-section sensitivity is
   the existing rate-term marginal, unchanged).
2. **Pareto constraint** — ACTIVE in spirit: the codec adds a rate-vs-pose-fidelity
   Pareto curve on the pose section; the default is the Pareto-dominant point. The
   net-S admission is gated on byte-closed exact eval (not the byte saving alone).
3. **bit-allocator hook** — ACTIVE: the pose section is now a tunable (rank, levels)
   byte knob; the default dominant point is the recommended allocation.
4. **cathedral autopilot dispatch** — N/A (no paid dispatch; default-OFF, advisory).
5. **continual-learning posterior** — N/A (no exact anchor yet; the durable measurement
   JSON + this memo are the bridge artifact until a byte-closed row lands).
6. **probe-disambiguator** — the codec ITSELF is the disambiguator between "iid is
   near-Pareto-optimal" and "low-rank helps": `lowrank_pose_section_fidelity` +
   `test_pareto_dominant_operating_point_exists_and_is_the_default` resolve it
   empirically on the real pose.

`council_predicted_mission_contribution`: `frontier_breaking_enabler` (a small honest
rate lever that folds into the bind-all archive grammar; net win pending exact eval).

## Files

- `src/tac/torch_vehicle/pose_film.py` (codec + auto-dispatch + guards)
- `src/tac/torch_vehicle/pose_film_v2.py` (re-export)
- `src/tac/torch_vehicle/driver.py` (config + validation + byte-close wire-in)
- `src/tac/torch_vehicle/tests/test_pose_lowrank_codec.py` (23 NO-FAKE tests)
- durable measurement: `.omx/research/pose_lowrank_CORRECTED_fidelity_20260617.json`
