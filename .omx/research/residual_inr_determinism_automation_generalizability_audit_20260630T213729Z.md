# Residual-INR hybrid pipeline — pre-fire rigor audit: determinism / automation / generalizability

- **Date:** 2026-06-30T21:37:29Z
- **HEAD:** a2460b14e (`v2 residual-only training mode + inflate residual-compose`)
- **Reviewer role:** senior reviewer, CPU-only. No GPU, no launch, live n600 baseline/dashboard untouched.
- **Authority:** `[macOS-CPU advisory] NON-PROMOTABLE`. The pointer is UNMOVED 0.19110; nothing here moves it.
- **Scope:** residual-only trainer mode (a2460b14e), `src/tac/v2_compose/`,
  `tools/compose_witness_archive.py`, `tools/launch_witness_run.py`, `tools/witness_autoconfig.py`.
- **Tests run (CPU):** `test_v2_residual_compose` (17) + `test_v2_compose_*` + `test_witness_autoconfig`
  + `test_launch_witness_run` = **84 passed**. Plus 3 bespoke empirical determinism/coherence probes (below).

## VERDICTS

| Axis | Verdict | Fire impact |
|---|---|---|
| 1 — Deterministic reproducibility | **PASS** (1 minor gap: provenance) | not a blocker |
| 2 — Automation | **GAP** | **BLOCKS the rate-bearing fire** (see G1) |
| 3 — Generalizability while contest-overfit | **PASS with notes** (seams, not defects) | not a blocker for the contest clip |

The single **fire-blocker is G1**: the *automated* entry point emits the **superseded** (non-rate-shrinking)
launch command and produces the **wrong npz format**, so firing the pipeline as-built would NOT test the
rate-bearing residual hypothesis. The determinism + math + parity foundations are solid.

---

## AXIS 1 — DETERMINISTIC REPRODUCIBILITY — **PASS**

### PASS evidence
- **Single recorded seed → all RNG seeded.** `experiments/train_levelset_witness_realized_through_R_mlx.py:826-827`
  (`mx.random.seed(args.seed)` + `np.random.seed(args.seed)`), `:985` (`np.random.default_rng(args.seed)`),
  `:1761` (`np.random.default_rng(seed+777)`); `--seed` default 0 (`:2313`). The Fourier basis uses a fixed
  `_FOURIER_SEED` (`experiments/train_witness_realized_through_R_mlx.py:116`). The trainer even persists the
  RNG bit-generator state (`...mlx.py:784` `__rng_hardness_json`) so a resume reproduces the draw sequence.
- **Resumable-from-disk + per-stage checkpoints.** Emitters always include `--stage-checkpoints`
  (`launch_command.py:157,278`; `witness_autoconfig.py:565`) and `--ckpt-every` (`witness_autoconfig.py:564`);
  `--resume-from` plumbed (`launch_command.py:164-165`). Satisfies the resumability non-negotiable.
- **numpy-fp32 is the authority; MLX is gradient-only.** The realized **verdict** path is numpy
  (`...mlx.py` `_render_for_pair_np` / `_fwd_numpy`, diff hunks at `:1341-1351`, `:1478-1490`), and the decode
  is the MLX-free numpy inflate (`archive_grammar.py:456-802`). MLX only produces the training gradient
  (`render_through_R_mlx`). This is the accepted design (CLAUDE.md: numpy-fp32 = bit-identical verdict
  authority, MLX never authority). **Accepted limitation:** MLX-GPU training is not guaranteed bit-identical
  run-to-run (GPU reduction nondeterminism); it changes the *trajectory*, never the *verdict/decode* — which
  is exactly why numpy is the authority. Not a gap.
- **Deterministic decode — verified bit-identical.** Empirical probe: built `archive.zip` twice, 1.2 s apart,
  from identical inputs → **identical SHA256** (`f9fb47a29db9fd56…`, 974 B both). Driver:
  `archive_grammar.assemble_v2_packet:383` pins `ZipInfo(date_time=(1980,1,1,0,0,0))` + `ZIP_DEFLATED`.
  Keyframes use a fixed LZMA filter chain (`contour_codec.py:43`, `FORMAT_RAW` + fixed preset/lc/lp/pb) →
  `store_blob` byte-deterministic (probe confirmed). Residual INR weights → int8-sym + brotli q=11
  (`archive_grammar.py:223-269`), all content-addressed.
- **Inflate == train, bit-exact (the NO-FAKE chain).** `test_v2_residual_compose.py:259`
  (`test_inflate_residual_compose_bit_exact_parity`) runs the *self-contained inflate.py as a subprocess*
  WITH a real per-class warp + residual and asserts `np.array_equal(raw, oracle)` vs
  `archive_grammar.residual_inflate_reference`. `test_v2_compose_archive_grammar.py:112` asserts the
  empty-residual floor is bit-identical to the proven `bulk_generator` path.
  `test_v2_residual_compose.py:292` confirms `compose_fn=None` is byte-identical to the bare render
  (the additive default-OFF safety claim). Strong.
- **Training-artifact npz is also byte-stable.** Probe: `residual_target.npz` written twice 1.2 s apart →
  identical size AND identical SHA256 (numpy `savez_compressed` pins a 1980 zip timestamp). So even the
  non-scored training artifact is byte-reproducible — better than the docstring claims.

### GAP A1.1 (minor, not a fire-blocker) — provenance is incomplete
The deterministic-reproducibility non-negotiable item (6) requires **git hash + upstream-snapshot-sha** with
every result. Grep across `tools/compose_witness_archive.py`, `tools/launch_witness_run.py`,
`tools/witness_autoconfig.py`, `src/tac/witness_autoconfig.py`, `src/tac/v2_compose/*.py`, and the trainer
found **no `git rev-parse` / git-hash / upstream-snapshot-sha recording**. What IS recorded: seed, config
snapshot, RNG state, archive sha256+size (`archive_grammar.byte_accounting:420`), advisory axis tags.
- **Fix:** stamp `git rev-parse HEAD` + `upstream/` snapshot sha into the trainer `result.json`, the
  `compose_witness_archive` phase reports, and the `launch.sh` header (`launch_witness_run.build_launch_sh:84`).
  Cheap; closes item (6).

---

## AXIS 2 — AUTOMATION — **GAP (fire-blocker G1)**

### PASS evidence
- **No-silent-failure launch.** `launch_witness_run.py` flag-validates every emitted flag against the real
  argparse and *refuses* before writing anything (`:196-205`), writes the command into `launch.sh` so the
  daemon argv is `["bash", launch.sh]` (no word-split fragility, `:77-98`), launches via
  `spawn_durable_daemon` which **auto-verifies the child survived exec** (`spawn_durable_daemon.py:205,215`;
  `--verify-s`), then **verifies the perf-env fast path** (`verify_perf_env:102`, warns loudly if
  `custom_grouped_backward` is INACTIVE — the ~17x silent-slow footgun) and **confirms the dashboard**
  (`ensure_dashboard:134`, which auto-tracks every new run each refresh tick). This is a genuinely strong
  one-command launcher for the *full-partition* witness.
- **Flag-validation is dogfooded** (`witness_autoconfig.py` CLI `:90-101`; `launch_command.parse_trainer_flags`
  statically greps the real `add_argument` lines — honors "NEVER invent CLI flags").

### GAP G1 (FIRE-BLOCKER) — the automated path does NOT emit the rate-bearing residual run
The trainer side of a2460b14e is correct and parity-tested, **but the composition tool was not updated to
match it.** Three coupled wiring defects mean "run the pipeline as documented" does **not** produce the
rate-shrinking run:

1. **Wrong emitter.** `tools/compose_witness_archive.py:149` calls **`build_residual_inr_command`** — the
   `--structured-init` path that *bakes the bulk INTO the INR weights = NO rate shrink* (its own
   `missing_capability_note`, `launch_command.py:201-208, 315-320`, says it is SUPERSEDED). The corrected
   emitter `build_residual_only_command` (emits `--residual-mode` + `--residual-target-npz`,
   `launch_command.py:228`) is **never called** anywhere (grep: only its def + `__all__`).
2. **Wrong / missing npz.** phase_a produces `residual_target.npz` via `save_residual_target`
   (`compose_witness_archive.py:116-118`) — the **diagnostic** floor (keys: `bulk_argmax_through_R`,
   `gt_lstars`, `residual_mask`). The `--residual-mode` trainer instead consumes the **training bundle** via
   `load_residual_training_bundle` (keys: `bulk_rgb_render_res`, `composition_mask`). The pipeline **never
   calls** `save_residual_training_bundle`. Probe confirms the formats are incompatible: feeding
   `residual_target.npz` to `load_residual_training_bundle` → `KeyError: composition_mask_shape`.
3. **No byte-close path for the real 4-section archive.** `phase_b` refuses `--residual-inr-weights` with a
   `SystemExit("NEEDS-WIRING")` (`compose_witness_archive.py:253-258`); it can only byte-close the
   deterministic FLOOR (empty residual). The building blocks exist and are tested
   (`archive_grammar.build_residual_blob` + inflate `has_residual` branch + parity test), but the tool glue
   (trained weights → `build_residual_blob` → 4-section archive) is unwired.

**Why this blocks the fire:** the binding sub-0.15 hypothesis is "a SMALL residual INR (bulk OUTSIDE the
counted weights) shrinks the rate." Firing the *emitted* command tests the SUPERSEDED `--structured-init`
full-weight INR (no shrink); manually switching to `--residual-mode` and pointing at the emitted
`residual_target.npz` KeyErrors. The operator could hand-assemble the correct run, but the "one automated
entry point" does not.
- **Fix (pre-fire):** in phase_a, (a) call `build_residual_training_bundle`/`save_residual_training_bundle`
  from the deterministic bulk render + warped labels (`bulk_generator.generate_bulk_render_and_labels` already
  produces both), (b) switch the emitted command to `build_residual_only_command` pointed at that bundle, and
  (c) wire phase_b to consume trained weights → `build_residual_blob` → the real 4-section archive. Then the
  parity test already in-tree guarantees inflate==train.

---

## AXIS 3 — GENERALIZABILITY (clip-agnostic machinery, contest-overfit data) — **PASS with notes**

### PASS evidence (the seam is held where it matters most)
- **The store/learn split SELF-DETECTS by measured signature — no class-index decision.**
  `store_learn_split.encode_known_split:228-262` decides GENERATE-vs-LEARN purely on `best_dseg <=
  recoverability_dseg_max` and `rel_improvement` thresholds over the **measured** per-class
  warp-through-R recoverability (`load_warp_recoverability_from_grok`). `cls_index` is carried as
  **metadata only** (`ClassAssignment.cls_index`, explicitly "never the decision basis", `:104-107`,
  `:206-207`). This is exactly the CLAUDE.md non-negotiable (self-detect, never hardcode the SegNet class
  index). The clip-agnostic rule is parameterized by the measured recoverability so the FORM transfers.
- **Config derivation is per-clip and provenance-tagged.** `witness_autoconfig.derive_config` runs value
  *generators* (TwoNN/MLE intrinsic-dim → Whitney mod-dim, RD → hidden-dim, annealing → curriculum) and a
  principled **portability split** (`Portability.SCORER_FIXED / DOMAIN / INSTANCE`, `:427-458`); fallbacks are
  flagged `SRC_FALLBACK` (NO-FAKE: a generator that can't measure never fabricates one).
- **The fitted physics IS derived, not baked.** Calibration `s_t/s_r/pitch` comes from the clip's
  `reach.json` (`bulk_generator.load_calibration_from_reach:364`); reach k* drives keyframe spacing
  (`load_reach_kstar`). The OVERFIT correctly lives in the DATA (keyframes, palette, pose scalars, residual
  weights), not the algorithm.

### NOTE A3.1 — phase_b re-hardcodes the per-class warp instead of using the derived plan
`compose_witness_archive.py:235` hardcodes `warp_codes = [0, 3, 2, 3, 1]` (by class position), whereas
phase_a *derives* it from the signature plan (`:129` `_WARP_NAME_TO_CODE[plan.per_class[c].warp_type]`). So
the byte-closed archive (phase_b) discards the generalizable derivation. For the contest clip the hardcoded
values match the derived ones, so the bytes are correct; on another clip phase_b would ship the wrong
routing, and it is a two-sources-of-truth smell. **Fix:** phase_b should reuse the phase_a plan.

### NOTE A3.2 — the inflate IGNORES the stored per-class warp mask (dead bytes + generalization seam)
The inflate parses `warp_codes` (`archive_grammar.py` inflate `:498`) but `_composite_warped` hardcodes the
routing — `fg = np.isin(cg, [0,1,3])` / `cr==2` / `ci==4` (`:591-592`) — and **never consumes `warp_codes`**
(grep confirmed). So the stored 5-byte warp-type mask is COUNTED-but-unused (a no-op-detector-class smell)
and, on a clip whose signature split differs, the stored codes are silently not honored. Faithful for the
contest clip (the parity test pins it to the canonical-order proven path). **Fix:** either consume
`warp_codes` in `_composite_warped` (true generalization) or drop the stored mask (reclaim the 5 bytes).

### NOTE A3.3 — frozen-net class indices + device intrinsics are hardcoded (defensible, document the boundary)
`residual_compose.LEARN_CLASSES=(1,3)` and `archive_grammar.BULK_IDX=(0,2,4)` hardcode the comma10k canonical
order. Per the CLAUDE.md class-order non-negotiable this order is **SCORER_FIXED** (the frozen SegNet always
emits it across the whole comma corpus), so hardcoding is defensible — but it is NOT wired from the
signature-derived split, so the two could silently disagree. Camera intrinsics
(`archive_grammar.py:467-469`, `NATIVE_FX=910`, `NATIVE_CX=582`, `NATIVE_CY=437`, `CAMERA_HEIGHT_M=1.22`) are
the comma-rig DOMAIN constants (same RAV4 rig per the corpus note); the per-clip calib that rides on them IS
fitted. These are device-portability boundaries, not contest-clip overfit. **Fix (later):** thread the
derived `learn_classes` from the split into `residual_compose`/`build_residual_blob` so there is one source
of truth; treat intrinsics as a per-rig DOMAIN constant block.

### NOTE A3.4 — advisory budget constants baked in a general module
`store_learn_split.py:70-72` bakes `_D_POSE_SIDECAR=3.4e-5`, `_BULK_DSEG_FLOOR=0.0185`, `_FRONTIER_S=0.19110`
as module constants used in the break-even **budget** report; the *actually measured* bulk floor is computed
live as `residual.bulk_dseg`. Advisory-only (0 archive bytes), but for a new clip the report's break-even
would use the stale 0.0185 rather than that clip's measured floor. **Fix:** feed the measured
`residual.bulk_dseg` into the break-even report.

---

## What is solid (do not re-litigate)
The math + parity foundations are genuinely strong: deterministic bit-identical archive, fixed-filter LZMA
keyframes, the inflate-as-subprocess bit-exact-vs-numpy-oracle NO-FAKE chain (with AND without residual), the
additive default-OFF residual mode proven byte-identical, fail-closed config guards
(`--residual-mode` requires the bundle; incompatible with `--structured-init`/`--lane-prior-phi1`/
`--freeze-decoder-fit-codes`; bundle-without-mode rejected — diff hunks `...mlx.py:2755-2789`), and a
signature-based (not index-based) store/learn decision. 84 CPU tests green.

## Bottom line for the operator
- **Determinism: PASS** (add git/upstream-sha provenance — A1.1).
- **Automation: GAP — G1 blocks the rate-bearing fire.** Wire phase_a to build the residual *bundle* + emit
  `build_residual_only_command` (`--residual-mode`), and wire phase_b to assemble the real 4-section archive,
  BEFORE firing — otherwise the automated launch tests the superseded non-shrinking run or KeyErrors.
- **Generalizability: PASS with notes** — the decision is signature-based (the hard part is right); reconcile
  the phase_b hardcode (A3.1) and the dead/ignored warp mask (A3.2) so the machinery is truly one-source-of-
  truth on the next clip. None of A3 blocks the contest-clip fire.
