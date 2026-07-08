# SAFE-COMPILE v7 — GPU per-chip BIT-IDENTITY cert at REAL witness coverage (#252) — 2026-07-08  [no-triality]

Crux-engineering task (operator, RESPAWN): *"All are fixable through crux engineering."* Produce the
missing **GPU per-chip bit-identity certificate** for the safe-compile regions so the hosc `_act` flip
becomes ADMISSIBLE-or-REFUSED **at v7 launch from evidence** instead of deferred to stop-time. Key
insight honored: bit-identity certs are **contention-INVARIANT** — measured safe beside the live run
(pid 63069, unharmed); NO speed/throughput benches run (those are contention-sensitive; out of scope).

**MEANS, not ends.** Pointer 0.19110 UNMOVED — this is compute-facet apparatus. A bit-identity
certificate is an on-device FACT ([macOS-MLX research-signal]), NEVER a score. Only a byte-closed
`upstream/evaluate.py` n600 exact row moves the pointer. Default `--safe-compile-regions none` remains
byte-identical; this cert only makes the OPT-IN flip evidence-gated instead of deferred.

## STORES CONSULTED
`.omx/research/mlx_safe_compile_v2_finish_20260708.md` (v2 landing: schema-v2 fingerprint, launcher b2,
device-bug fix, the CPU 5.96e-8 hosc failure + coverage-limited-n=32 residual) · `src/tac/mlx_safe_compile.py`
(harness/manifest/resolve/install) · `src/tac/tests/test_mlx_safe_compile.py` · `.omx/state/mlx_safe_compile_manifest.json`
(the CANONICAL evidence surface — was v1, fingerprint-ABSENT, toy (256,96)/beta=2.5 coverage) ·
`.omx/research/t5_crucible/position_INCL_S4_rudin.md` D-2 (the failing 1/9 CPU region IS the flip lever;
GPU manifest fingerprint-ABSENT; v1 GPU hosc cert n=32 coverage-limited) ·
`experiments/train_levelset_witness_realized_through_R_mlx.py` (`_act` @1155: `tanh(hosc_beta*sin(hosc_omega*u))`,
u = (P, hidden_dim=96) trunk pre-activation; `--hosc-beta 4.0`, `--hosc-beta-end` anneals up to ~10) ·
`tools/launch_witness_run.py` b2 (manifest freshness admission) · live run
`levelset_n600_crucible_v6_run1` pid 63069 (GPU-occupied, ~25–62 GB; untouched).

## WHAT CHANGED (crux engineering — closes the two v2 residuals for the flip lever)
The v2 residuals blocking admissibility were **(a) canonical manifest fingerprint-ABSENT + schema v1**
(so per-chip trust could not actually enforce), and **(b) coverage-limited cert** (toy (256,96) shape at
a SINGLE beta=2.5 — "n=32 can pass while unsampled inputs diverge"). Both fixed in
`src/tac/mlx_safe_compile.py`:

1. **REAL witness coverage.** The activation/tail region `make_inputs` now generate at the **real 384×512
   render grid** (`_CERT_GRID_POINTS = 196_608`) × `hidden_dim=96` — the actual per-frame pre-activation
   shape `_act` sees — with ADVERSARIAL value coverage of the fp-contraction knife-edges (`_real_coverage_array`:
   dense linspace across sin's steep zero-crossings in col 0 + large-magnitude saturation probes driving
   `beta·sin` onto tanh's rails).
2. **Beta swept across the v7 anneal range.** `_sweep_hosc_beta` maps the harness's 32-input seed block
   (`1000..1031`) via `seed % 32` — a permutation of `0..31` — so the canonical cert **tiles β∈[1.0, 10.0]
   UNIFORMLY with BOTH endpoints hit** (β=1.0 and the step-native β=10.0, the hardest saturation case).
   Not a random hash: the anneal endpoints are guaranteed in the sweep.
3. **CLI `--reps`** so certification runs with minimal timing reps (good GPU citizen beside a live run;
   bit-equality is contention-invariant, timing is not a certificate). The activation `fn` bodies are
   UNCHANGED — `_pure_hosc` still matches the wired region exactly; only the certification input
   distribution improved.

The canonical manifest was re-certified on THIS chip's GPU via the module's canonical write path
(`python -m tac.mlx_safe_compile --certify --device gpu --n-determinism 5 --reps 2 --out .omx/state/mlx_safe_compile_manifest.json`
→ `CertificationManifest.save`), now **schema v2, device=gpu, fingerprint {Apple M5 Max / 25E246 / 0.31.2}**.

## CERT TABLE (region × device × max|Δ| × N cross-process × coverage) — measured on M5 Max GPU
| region | device | verdict | max\|Δ\| | N (cross-proc) | input coverage | shape |
|---|---|---|---|---|---|---|
| **hosc_activation** (THE flip lever) | gpu | **CERTIFIED** | **0.0** | **5/5** | 32 inputs, β∈[1,10] uniform | (196608, 96) |
| siren_activation | gpu | CERTIFIED | 0.0 | 5/5 | 32 | (196608, 96) |
| wire_activation | gpu | CERTIFIED | 0.0 | 5/5 | 32 | (196608, 96) |
| finer_activation | gpu | CERTIFIED | 0.0 | 5/5 | 32 | (196608, 96) |
| film_modulate | gpu | CERTIFIED | 0.0 | 5/5 | 32 | (196608, 96) |
| film_affine_tail | gpu | CERTIFIED | 0.0 | 5/5 | 32 | (196608, 96) |
| sigmoid_scale | gpu | CERTIFIED | 0.0 | 5/5 | 32 | (196608, 3) |
| compose_rgb_tail | gpu | CERTIFIED | 0.0 | 5/5 | 32 | (196608, 3) |
| achromatic_luma | gpu | CERTIFIED | 0.0 | 5/5 | 32 | (196608, 3) |
| ce_reduction | gpu | CERTIFIED (fixed-order) | 0.0 | 5/5 | reduction, determinism-only | — |

`kernel_candidates == []` (zero failures on GPU — the D16 fp-contraction-kernel pipeline stays dormant here).

**Timing is NOT a deliverable** (out of scope; contention-sensitive). The manifest's `speedup` column is a
harness byproduct at `--reps 2`, non-authoritative — the finished GPU whole-step B=8 number remains the
v2 PIECE-3 run-1-stop deliverable.

## ADMISSIBILITY VERDICT — the hosc `_act` flip is ADMISSIBLE at v7 launch (GPU)
Measured `resolve_enabled_regions('hosc_activation', manifest, run_device='gpu')` → `{hosc_activation}`
(**ADMIT**); `run_device='cpu'` → `∅` (**REFUSE** — device mismatch, correct). Launcher b2
`manifest_fingerprint_ok` → True on this host. So a v7 GPU launch arming `--safe-compile-regions
hosc_activation` will pass b2 and `resolve_enabled_regions`, and the flip activates from EVIDENCE
(evidence replaces the D17 deferral).

**verdict_scope (honest bounds).** This is a POSITIVE cert, scoped to: **{chip=Apple M5 Max, macOS=25E246,
mlx=0.31.2, device=gpu}**, region=`hosc_activation` (and 8 sister SAFE regions + the fixed-order reduction).
It does NOT generalize off-chip/off-device/off-mlx-version — the manifest is fingerprint+device stamped
and `resolve_enabled_regions` fails closed on any mismatch (a different host recertifies). It vindicates the
per-chip + per-device design: the SAME hosc activation **REFUSES on CPU** (measured 5.96e-8 fp-contraction,
1 ULP, v2 memo) but **ADMITS on this GPU** at full β∈[1,10] uniform coverage over the real 384×512 grid,
N=5 cross-process bit-identical. The certificate is an EMPIRICAL bit-identity fact over a finite (now
real-witness-scale, adversarial, endpoint-covering) input sample — strong evidence, not a symbolic proof;
a future MLX/macOS bump re-arms the cert.

## WHAT LANDED
- `src/tac/mlx_safe_compile.py`: `_CERT_GRID_POINTS`/`_CERT_HIDDEN`/`HOSC_BETA_ANNEAL_RANGE`,
  `_real_coverage_array`, `_sweep_hosc_beta` (uniform-tile β∈[1,10]); all activation/tail builder
  `make_inputs` → real grid coverage; CLI `--reps`; `HOSC_BETA_ANNEAL_RANGE` in `__all__`.
- `src/tac/tests/test_mlx_safe_compile.py`: +4 tests (β sweep endpoints/uniformity · real-coverage grid
  shape + adversarial band [gpu] · hosc GPU real-coverage bit-identical [gpu-skip] · GPU-device
  fingerprint-stamped manifest roundtrip + ADMIT/REFUSE resolve). 49 pass; ruff F clean.
- `.omx/state/mlx_safe_compile_manifest.json` (untracked host-state): re-certified schema-v2, device=gpu,
  fingerprint-stamped, all 10 CERTIFIED at real coverage.

No config flag was flipped (the v7.3 compiler consumes the evidence). No training launched; no heavy jobs.
Code carries its triality treatment (DSL lever `SafeCompileRegions` from v1; no new equation — compute-facet
apparatus). This memo = [no-triality].
