# Y1 — G3 readiness harness now byte-closes + evals FiLM-v2 checkpoints

**Authority:** `[contest-CPU advisory]` NON-PROMOTABLE. $0 / CPU + code only. No GPU touched;
the running arm_b job (`launch_bind_all_taper_ab … --go`, pid ~89235) was not disturbed.
No score / frontier / promotion claim is made.

**Date:** 2026-06-16T23:53:20Z. **Lane:** the G3 exact-row readiness actuator
(`tools/verify_e2e_byte_close_eval.py`). Sister of the P3 finding.

## What was broken (the P3 bug)

`tools/verify_e2e_byte_close_eval.py` is the readiness actuator that turns a converged
small-basis `best/` checkpoint into a byte-closed contest `archive.zip` + an advisory `S`
(the input the G3 dual exact row consumes on contest Linux x86_64 CPU / CUDA).

Its byte-close + parse-back-parity logic is sound, but the decoder construction assumed a
**plain vendored decoder state-dict**:

* `_build_decoder` built a bare `HNeRVDecoder` / `ConfigurableTaperHNeRVDecoder` and
  `load_state_dict(dec_sd_back, strict=False)`.
* `_byte_close_and_verify_parity` called `vb.build_archive(dec_sd, latents, meta_dict)`
  directly on the raw state-dict.

The **arm_b production run is FiLM-v2** (`launch_bind_all_taper_ab.py`:
`pose_film_enabled=True, pose_film_version=2`). Its `best/best_ema_decoder.pt` is the
**WRAPPER** state-dict, not a plain decoder:

```
decoder.<vendored-key>   # the inner HNeRVDecoder, under the `decoder.` prefix
pose_mlp.fc1.weight ...  # the v2 pose-conditioning MLP
film_resid.proj.weight … # the v2 residual-rgb0 FiLM
stored_pose              # the (n_pairs, 6) GT-pose buffer (range-coded ~bytes at byte-close)
```

So the harness errored: `vb.build_archive` does not understand `decoder.*` / `pose_mlp.*`
keys, and the eval-decoder load mismatched (`stem.weight` absent → base_channels could not
be inferred; `stored_pose` is not a decoder key). The G3 actuator could not process the
arm_b output.

## The fix — REUSE the driver's PROVEN FiLM-v2 export path (no codec reimplemented)

The driver's `_build_archive_and_eval_decoder` (`src/tac/torch_vehicle/driver.py:2287`)
already has the FiLM-v2 branch and is the in-loop byte-close arm_b uses every BEST snapshot.
The harness now AUTO-DETECTS a FiLM wrapper and routes through the SAME canonical helpers:

* `tac.torch_vehicle.pose_film_v2.build_archive_with_pose` — vendored 3-section archive +
  the additive ~1 KB PFLM pose section (vendored codec stays PRISTINE; pose section appended).
* `pose_film_v2.parse_pose_section` — reads the additive pose section back.
* `pose_film_v2.wrapper_sd_to_archive_decoder_sd` — splits the wrapper sd into the codec
  decoder blob (bare vendored keys + FiLM weights) + drops `stored_pose` (→ pose section).
* `pose_film_v2._FiLMEvalDecoder` + `PoseFiLMHNeRVWrapperV2` — the cursor eval adapter that
  renders the SAME FiLM-conditioned frames `inflate.py` produces.

These are the EXACT modules `driver._build_archive_and_eval_decoder` imports. v1
(`pose_film.*`) is also handled via `_FILM_PARAM_PREFIXES` (mirrors the driver) so the route
is version-aware; the plain-decoder path is unchanged (backward-compatible).

### New harness surfaces (`tools/verify_e2e_byte_close_eval.py`)

| function | role |
|---|---|
| `_detect_film_version(dec_sd)` | `None` plain / `1` v1 / `2` v2 — keys on `stored_pose` + the FiLM prefix set (same signature the driver uses) |
| `_infer_film_hidden(dec_sd, ver)` | cond_dim from `pose_mlp.fc1.weight` (v2) / `pose_film` MLP (v1) so the rebuilt wrapper is bit-identical |
| `_infer_dims` (extended) | base_channels now also reads `decoder.stem.weight` (the wrapper inner-decoder prefix) |
| `_byte_close_and_verify_parity_film` | additive-pose byte-close + parse-back parity INCLUDING the `stored_pose` section fixed-point |
| `_build_film_eval_decoder` | rebuilds the wrapper + `_FiLMEvalDecoder` from the parse-back blob + parsed pose (driver step 3) |
| `run()` (branched) | detect → FiLM route vs legacy route; report gains `pose_film_version` / `pose_film_hidden` |

`--taper-channels` is supported under FiLM-v2 (the v2 wrapper wraps a
`ConfigurableTaperHNeRVDecoder` inner decoder — `_build_film_eval_decoder` builds it via the
existing `_build_decoder(taper_channels=…)`).

## NO-FAKE parity proof

Real arm_b `best/` does not exist yet (the run just started), so the test builds a MINIMAL
FiLM-v2 wrapper checkpoint fixture from REAL modules (`HNeRVDecoder` +
`PoseFiLMHNeRVWrapperV2`) with identity/zero-init FiLM. Proven (all REAL, no stubbed codec):

* **byte-close + parity_ok=True** on a FiLM-v2 wrapped checkpoint, INCLUDING
  `pose_section_present` + `pose_section_fixed_point` (the additive pose section round-trips
  bit-exact). The parse-back blob carries the bare vendored keys + the FiLM keys (no
  `decoder.` prefix, no `stored_pose` — that lives in the pose section).
* **eval decoder rebuilds + renders** `(B, 2, 3, 384, 512)` from the parse-back blob + parsed
  pose (the cursor adapter the exact eval consumes).
* **identity-init render bit-equality**: an identity/zero-init FiLM-v2 wrapper renders
  BIT-EQUAL to the plain vendored decoder on BOTH heads (`f1` always seg-clean / pose-invariant;
  `f0` bit-equal at zero-residual init) — the NO-FAKE fidelity contract.
* **full `run()` end-to-end** (scorer stubbed; the byte-close/packet path is REAL) reports
  `pose_film_version=2, pose_film_hidden=8`, `parity_ok`, `pose_section_fixed_point`, and a
  runnable contest `submission_dir` (`archive.zip` holds ONLY `0.bin`).
* **backward-compat regression**: the plain n600 basin still byte-closes to **89,136 B /
  parity_ok=True** and reports `pose_film_version=None` (the pre-existing tests + the new
  CLI assertion).

`src/tac/tests/test_verify_e2e_byte_close_eval.py`: 13 tests pass (6 new FiLM-v2 + the
7 pre-existing plain-decoder regressions). Ruff clean.

## Outcome

The G3 readiness actuator now handles arm_b's FiLM-v2 output: when arm_b lands a converged
`best/`, `verify_e2e_byte_close_eval.py --ckpt-dir <arm_b>/best [--taper-channels …]` will
byte-close it (vendored archive + additive pose section), verify parse-back parity (incl. the
pose section), assemble the contest packet, and report the advisory `S` from the REAL
`archive.zip` `st_size` — the input to the G3 dual exact CPU/CUDA row. `[contest-CPU advisory]`.
