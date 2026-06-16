# G2 — C8 bilinear-skip ARCHIVE BYTE-CLOSE + oracle-parity gate: VERDICT

**Subagent:** G2 (DAG node, `grand_symposium_smallbasis_dag_verdict_20260616.md`).
**Date:** 2026-06-16T16:00:46Z. **Authority:** all numbers `[advisory]` (local CPU; NO MPS; NO paid/remote eval). **Mission tag:** frontier_breaking_enabler (unblocks the small-basis → exact-row path).

## HEADLINE VERDICT

**The small-basis (taper) HNeRV decoder IS byte-closeable. The claimed C8 blocker — "bilinear-skip archive export is a `NotImplementedError` / bilinear-skip CANNOT be byte-closed" (symposium verdict, frontier-ledger #81) — is FALSE / outdated.** The bilinear-skip is NOT in the codec or the archive grammar at all; it lives entirely in the decoder `forward()`, which the eval/inflate path runs natively via `F.interpolate(mode='bilinear', align_corners=False)`. The vendored codec serializes weights by name/shape/scale and is **architecture-/schedule-agnostic** — it neither knows nor cares that the forward contains a bilinear-skip.

There is **no `NotImplementedError`** anywhere on the byte-close path (`grep NotImplementedError src/tac/torch_vehicle/driver.py` → 0 matches; 0 in `src/tac/capstone_vq_nerv/`). The byte-close was already wired and the bilinear-skip already round-trips. I PROVED this empirically (3 exactness proofs + 2 parity proofs below), not by trusting docstrings.

**The unblock for the DAG: launch the small-basis run, byte-close `best/best_archive.bin` via the existing path, and proceed to G3 (dual CPU/CUDA exact row). No new export code is required for bilinear-skip.**

## SEARCH-FIRST inventory (read, cited file:line)

- **Vendored decoder** `…/hnerv_muon/src/model.py:42-54` — `forward()` bilinear-skip at L47-50: `identity = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)` → `skip(identity)` → `x = sin(self.ps(block(x)) + identity)`. Standard torch op.
- **Vendored codec** `…/hnerv_muon/src/codec.py:145-180` — `build_archive(decoder_state_dict, latents, meta_dict)` / `parse_archive(...)`. Stores: meta(brotli) + decoder(int8 per-tensor symmetric quant → zigzag → brotli q11) + latents(uint8 delta → zigzag → lo/hi split → brotli). **Iterates `sd.items()` by name/shape** (`encode_decoder` L59-71) — fully architecture-agnostic. Docstring L14: "Round-trip verified bit-exact."
- **Vendored inflate** `…/hnerv_muon/inflate.py` (66 LOC, torch) — `parse_archive` → rebuild `HNeRVDecoder` → `decoder(latents)` (runs bilinear-skip) → bicubic upsample to 874×1164 → uint8. `inflate.sh` is the standard contest runtime. **The inflate is torch-based and the bilinear-skip "just works" as the torch forward.**
- **Driver byte-close** `src/tac/torch_vehicle/driver.py:1561-1608` `_build_archive_and_eval_decoder` → calls `self.v.build_archive(ema_sd, ema_latents, …)` and `self.v.parse_archive(...)`, then `self._new_vendored_decoder(...)` (the torch decoder with bilinear-skip) for eval. `best_archive.bin` written at L1754. Full eval forward incl. bilinear-skip at L2286-2316.
- **Small-basis vehicle** `src/tac/torch_vehicle/configurable_taper_decoder.py:122-187` `ConfigurableTaperHNeRVDecoder` — verbatim vendored `__init__` + `forward` (bilinear-skip at L179), only the 7-stage channel schedule is configurable. Docstring L19-23: "bit-identical forward … round-trips through the schedule-AGNOSTIC vendored codec … unchanged." 19 parity tests pass.
- **Sister portable path** `src/tac/local_acceleration/pr95_hnerv_mlx.py:1008` `bilinear_resize2x_align_corners_false_nhwc` (0.0 drift vs torch) + `src/tac/capstone_vq_nerv` numpy-portable inflate (`capstone_hinerv_skip_gridpe_upgrade_20260611.md` L22,40: "bilinear-skip … ALREADY structurally present … pure-numpy inflate, op-for-op"). So a numpy-portable inflate of the bilinear-skip ALSO already exists in the capstone vehicle.
- **`#81` / "NotImplementedError"** is NOT a grep-able ledger id or a code symbol; it is a DAG-node label whose substantive claim lives only in `grand_symposium_smallbasis_dag_verdict_20260616.md:51-54`. That claim is contradicted by the code + the proofs below.

## WHY the blocker claim was wrong (root cause)

The symposium conflated two unrelated things: (1) the **FP4 row is NO-GO** (its own smoke: ΔS +0.25, d_seg +56%) — TRUE, and (2) "bilinear-skip cannot be byte-closed" — FALSE. The bilinear-skip is a `forward()` op, not an archive section; the codec never touches it. The likely origin of the myth: a numpy-portable-inflate concern (can pure numpy reproduce bilinear?) was mis-stated as an export blocker. Both halves are answered NO-BLOCKER below.

## PROOFS (empirical, `[advisory]`, local CPU)

Small-basis vehicle: `ConfigurableTaperHNeRVDecoder`, base_ch=20, `dseg_aware_taper` = `[16,16,17,19,19,14,10]`, 83,422 params, archive **84,885 bytes**.

**Proof A — byte-close round-trip reproduces frames (sub-quant):** trained(random-init) torch decoder forward (incl. bilinear-skip) → `build_archive` → `parse_archive` → rebuild taper decoder → forward → bicubic→uint8 camera frames vs reference: **max abs uint8 diff = 1, mean abs = 0.05, 5% pixels differ.** The ONLY divergence is int8 weight-quant rounding (≤1 LSB) — the codec's by-design lossiness, NOT a bilinear failure. Both sides ran the bilinear-skip identically.

**Proof B — codec is fixed-point (EXACT):** parse-back of a parse-back is **BIT-EXACT** on both weights (`torch.equal` all tensors True) and latents (True). The codec round-trips its quantized state exactly; once on the int8 grid the bytes are a fixed point.

**Proof C — build_archive is deterministic:** byte-identical on identical inputs (`arc_a == arc_b` True, 84,883 bytes).

**Proof D — frame determinism (the bilinear-skip is deterministic on byte-closed weights):** two independently-constructed parse-back decoders produce **BIT-IDENTICAL float frames** (`torch.equal(fa, fb)` True), including the bilinear-skip op.

**Proof E — bilinear IS scorer-identically numpy-portable:** pure-numpy `align_corners=False` bilinear (general gather form) vs torch `F.interpolate` across EVERY decoder upsample resolution (6×8→12×16 … 192×256→384×512): **max abs ≤ 3.6e-7** (float32 ULP-level, ~6 orders below the uint8 step 1.0). So a numpy-only inflate (no torch dep) would also reproduce the decoder scorer-identically — the bilinear-skip is portable on BOTH the torch axis (exact) and the pure-numpy axis (≤3.6e-7).

**Existing tests:** `pytest src/tac/torch_vehicle/tests/test_configurable_taper_decoder.py` → 19 passed (the codebase's own bit-identical-forward + codec-round-trip parity contract).

## VERDICT

| question | answer |
|---|---|
| Is the small-basis decoder byte-closeable? | **YES.** Proofs A-D. No new code needed; the wired path (`_build_archive_and_eval_decoder`) already does it. |
| Does bilinear-skip break byte-close? | **NO.** It is a `forward()` op, not an archive section; the codec is schedule-agnostic. |
| Is there a `NotImplementedError`? | **NO.** 0 matches on the byte-close path. |
| Is a numpy-portable inflate possible (if torch inflate is ever disallowed)? | **YES** — bilinear matches torch to ≤3.6e-7 (Proof E); capstone already ships one. |
| Does the small basis reach sub-0.15 once byte-closed? | **OUT OF SCOPE for G2** (that is G1 capacity-RD + the long train). At int8, even d_seg=0.0006 → 0.152 (above T_3) per the symposium; FP4 is NO-GO. G2 only proves byte-closeability, which is now PROVEN. |

## The ONE real remaining wiring gap (NOT a bilinear blocker)

The driver writes a per-video `best/best_archive.bin`; the contest packet needs an `archive.zip` containing the `.bin` files + an `inflate.sh`/`inflate.py` runtime. The vendored runtime (`hnerv_muon/inflate.{py,sh}`, 66 LOC torch) IS that runtime and is reused as-is for the taper decoder (same `HNeRVDecoder` import; the taper schedule rides in `meta`). The remaining work for G3 is the standard zip-packaging step (assemble `archive.zip` = {`0.bin`, `inflate.sh`, `inflate.py`, `src/`}) — a packaging task common to every substrate, not a bilinear/byte-close blocker. Minor sister gap: `tac.torch_vehicle.run` argparse does not yet expose `--taper-channels` (config-only); add the flag so a small-basis run is launchable from the CLI.

## Wire-in (6-hook) + housekeeping

- #1 sensitivity-map: N/A (this is an unblock proof, not a new score signal).
- #2 Pareto: N/A. #3 bit-allocator: N/A.
- #4 cathedral autopilot: N/A (no new dispatchable candidate; unblocks existing G3 path).
- #5 continual-learning: this memo + the parity test are the durable artifacts; the symposium's #81 blocker row should be superseded (APPEND-ONLY) by this MEASURED refutation.
- #6 probe-disambiguator: this memo IS the disambiguator between "bilinear blocks byte-close" (FALSE) and "int8 quant + FP4 NO-GO bound the SCORE" (the real, separate constraint).
- Durable test: `src/tac/torch_vehicle/tests/test_bilinear_skip_byte_close_g2.py` (NO-FAKE: asserts Proofs A-E on a real taper decoder).

## NO score claim

Nothing here is a score. All rows `[advisory]` local CPU. No paid/remote eval dispatched. The frontier is 0.191 and UNMOVED; G2 removes the (phantom) blocker so G3 can produce the first byte-closed dual CPU/CUDA exact row for this vehicle.
