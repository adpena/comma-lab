# ddm_hb2 — HPAC self-compress pack round-trip failure: diagnose + fix + rerun tq1c stages 3-4

**Fire-order (2026-08-08):** the hb1 driver (`/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/`)
completed tq1c stage-2 training rc=0 at 08:01:53Z, then stage-3 pack FAILED in 4s:
`RuntimeError: self-compressed round trip changed logits by 0.25`
(`pack_hpac_self_compress.py:269`, `--require-exact` class check — fail-closed by design, the
right behavior). The driver `continue`d to the gt arm per its loop; tq1c stages 3-4 are OWED.

**Measured state (recall-first, do NOT re-derive):**
- tq1c ep60 endpoint: bpp 0.006555495669264903 · top1_err 0.001652891370985243 ·
  token_bytes 96,665 · model_bytes 18,753 · **joint 115,418 B** · phase discrete_qat.
- **bit_depth_histogram ep60: {0: 21, 2: 9, 3: 16, 4: 38, 5: 112, 6: 225, 7: 89, 8: 7}** —
  heterogeneous, incl. 21 PRUNED (0-bit) channels and 9 at 2 bits.
- gt arm epoch 0 histogram: {8: 517} (uniform 8-bit fresh start) — PR130's own lineage likely
  never exercised the 0-bit / 2-bit serializer edges. 0.25 = 2^-2 = one quantization step at
  2 bits (or a scale-exponent clamp artifact at weight-exponent-min -6).
- Stage-2 vs stage-3 driver args MATCH on all architecture flags (channels 64, patch 64,
  delta 2, frame-dim 8, bounds 127/127, weight-exponent-min -6) — NOT an arg-mismatch bug.
- The round-trip check uses seeded torch.randint token inputs → the failure is structural
  (input-independent), not data-dependent.
- Checkpoint: `/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/checkpoints/tq1c/hpac_selfcompress_e60.pt`
- Code: `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/code/`
  (pack_hpac_self_compress.py · train_hpac_self_compress.py · codec_hpac_integer.py). The
  repro_repo is its OWN git repo on the SSD — commit fixes there normally (bare git OK, it is
  not the main pact tree), AND mirror the diff as a patch file in the findings dir committed
  to pact main via serializer.

**Deliverables:**
1. REPRODUCE the failure standalone (same argv as driver stage-3, seeded — should be exact).
2. BISECT the mechanism: per-module, per-channel serialize→deserialize comparison on the REAL
   checkpoint (compare `source` post-`set_deployed_bit_depths(…, True)` weights vs `restored`
   weights tensor-by-tensor; find WHICH channels differ and at WHAT bit depth). Name the
   mechanism precisely: 0-bit channel handling vs low-bit rounding tie vs scale/exponent
   mismatch between `enable_self_compression`-wrapped source and plain restored model
   (`model_from_args(args, False)` asymmetry is a named suspect — serialize path runs on the
   self-compression-enabled module, restore path on the plain module).
3. FIX in repro_repo code — smallest correct change; PR130 off-the-shelf grant covers reuse
   (memory `pr130_code_off_the_shelf_authorized_20260806`), honesty half UNCHANGED: the
   findings must carry a borrowed_substrate_accounting note (what is theirs, what we changed,
   why). The fix must keep the check FAIL-CLOSED (max_diff != 0.0 → raise) — cure the
   serializer, NEVER weaken the gate (memory: batch_shape_is_part_of_the_forward_instrument —
   repair = rebuild under the consumer's instrument, never loosen).
4. RERUN tq1c stage-3 (pack → hpac.bin.xz + report) then stage-4 encode AND decode with
   `--require-exact` per the driver's stage-4 block (same argv). Emit both reports.
5. FINDINGS: `.omx/research/ddm_hb2_20260808/HB2_FINDINGS.md` + typed receipts JSONL
   (mechanism row, per-channel diff summary, fix diff SHA, rerun rc + report paths, packed
   bytes vs the 18,753 estimate) + the patch file. Note for the gt arm: same driver will hit
   stage-3 at its ep60 (~1-2 days) — the fix lands in shared code so it inherits automatically;
   say so explicitly in findings.

**Boundaries:** CPU-only (OMP/MKL 4 threads max), NO Metal, NO scorer slot needed. Do NOT
touch `checkpoints/gt/` or kill driver pid 9316 (gt stage-2 training is LIVE under it). Do
NOT modify the driver script while it runs. Read-only toward all other live run dirs
(ARM-VEH n32 under ddm_mx1e_20260807 is LIVE on Metal).

## OPTIMAL FORM

This arm is a DIAGNOSIS + REPAIR of an existing mechanism at its family's REFERENCE form,
not a build or race of a reduced variant. Reference form = PR130's own self-compress pack
pipeline (repro_repo `pack_hpac_self_compress.py` + `train_hpac_self_compress.py`, provenance
pin: repro_repo git HEAD at arm start; checkpoint pin: `hpac_selfcompress_e60.pt` under
`checkpoints/tq1c/`, driver-recorded rc=0). ZERO scope reduction (the failing round-trip is
the FULL model, all 517 channels) and ZERO mechanism reduction (the fix must preserve the
exact-equality gate and the full serialize format). Any repair that narrows the gate,
special-cases the failing channels out of the check, or skips low-bit-depth channels is a
MECHANISM reduction and FORBIDDEN — that would be a toy verdict on the pack family.

**Discipline:** pact-side artifacts commit via `tools/subagent_commit_serializer.py` with
POST-EDIT `--expected-content-sha256` per file; tags `[no-triality] [p0-ledger-ok]`;
review_tracker ×2 per pact .py (repro_repo .py edits are outside the pact review gate — still
self-review rigorously); NO Claude/AI attribution or Co-Authored-By trailer anywhere (both
repos). If serializer hits sandbox git-perms, write artifacts + say so.
