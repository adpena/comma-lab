# ddm_sd1 — THE DISTORTION AXIS on PR130's base: the semantic renderer's real R-D curve

Everything banked today is RATE. Nobody has touched PR130's DISTORTION axis on its own base.

## BINDING STATE (measured, do not re-derive)

BASE = **PR130 CPR1**, `S = 0.172141297491896447` `[contest-CUDA, DALI GT, n600]`, archive 191,052 B,
sha `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`.
Reproduced byte-identically at `/Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/reproduction/`.
Decomposition: seg 0.028609 (16.62%) · pose 0.014025 (8.15%) · **rate 0.127214 (73.90%)**.
Sub-0.15 by rate alone needs **−33,252 B (−17.4%)**.

Archive anatomy (MEASURED from the real archive, not a memo):
```
archive.zip 191,052 = ZIP overhead 100 + member `p` 190,952
p          = [u32 models_bytes][ LZMA(models_raw) 73,968 ][ HPAC tokens 116,980 ]
models_raw = 83,493 = 8 + semantic 40,252 + pose_carrier 23,054 + hpac_weights 20,179
```

BANKED THIS SESSION (both scorer-free, both RECEIVER-BLOCKED):
- **split-stream model pack −903 B** — 3 separate brotli-q11 streams instead of one joint LZMA.
  Real archive built (190,149 B), parse-back EXECUTED and byte-identical.
  `.omx/research/ddm_pr130_reproduce_20260809/SPLITPACK_REAL_ARCHIVE_ROW.md`
- **ANS token coder ≈−2,080 B** — PR130's `queue.RangeEncoder` runs +1.8809% above its own model's
  entropy (MEASURED on real AR tables); `stack.AnsCoder` runs +0.0559%. Recovers 97.7% of the gap.
  `.omx/research/ddm_pr130_reproduce_20260809/ANS_REAL_TABLE_MEASUREMENT.md`

MEASURED NEGATIVES (do NOT re-run):
- Whole-section coder race on `semantic`: brotli-q11 wins; 11 alternatives lose by 413–3,754 B; all
  L21 byte-maps (off/zigzag/delta) lose or tie. **The whole-section coder axis is SHUT.**
- Generic coders on the token stream: brotli +5 B, lzma2 +64 B. Shut.
- `Categorical(perfect=True)`: saves 0 B. Refuted.
- Per-frame encoder flush: refuted at source (one RangeEncoder outside the loop).
- clean60 (60ep HPAC retrain on our labels) LOST to PR130 by +185 B apples-to-apples.

## HARD CONSTRAINTS
- `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/` is **READ-ONLY**. Never edit, never
  `git add` inside it. Import from it; copy out to build on.
- The pinned `upstream/` snapshot is IMMUTABLE.
- Always `.venv/bin/python`. Bulk artifacts → `/Volumes/VertigoDataTier/pact/`. Never `/tmp` in
  persisted evidence.
- Commit via `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256` per
  file, tags `[no-triality] [p0-ledger-ok]`. NO Claude/AI attribution, NO Co-Authored-By.
  `REVIEW_GATE_OVERRIDE=1` is FORBIDDEN with `.py` files — use
  `tools/review_tracker.py mark-file <f> --status reviewed` twice per `.py`.
- MPS/MLX are NEVER score authority. Label `[macOS-CPU advisory]` / `[macOS-MLX research-signal]`,
  `score_claim=false`. Only `upstream/evaluate.py` on exact archive bytes is a score.
- Reuse RECORDED argv verbatim where one exists (e.g.
  `ddm_pr130_encode_tokens_metal_20260809/run/launch_manifest.json`) rather than retyping flags.
  NEVER invent CLI flags — grep `add_argument` first.
- Every number MEASURED or labelled DERIVED/PROJECTED. A projection is never reported as a measurement.
- ONE Metal fire at a time; heavy launches through the governed path.

## WHY THIS IS TRACTABLE LOCALLY

The semantic leg is **REPRODUCED at inference on Metal** — `quant_bits=4`, 66,339 params, through
`train_semantic_quantized.evaluate_all`: DALI-GT **0.0002857038709852431** (0.998650× published),
AV-GT 0.0002764044867621528, **19 s/n600**. So real d_seg is measurable here, cheaply, without CUDA.
You own the scorer slot.

The semantic renderer is **36,580 B packed (19.15% of the archive)** and it OWNS the entire seg term
(0.028609 = 16.62% of S). Tokens are lossless by construction and carry ZERO distortion coupling —
so the seg/rate tradeoff lives entirely here.

## THE TASK

Trace the semantic renderer's real (bytes, d_seg) curve on PR130's base and find whether the shipped
point is on the joint optimum:
- quantization depth (it ships `quant_bits=4` — is 3 or 5 better in S units?),
- capacity/width, if the checkpoint's config exposes it,
- per-tensor bit allocation by measured seg-sensitivity.

Score every candidate in **S units**, not in d_seg or bytes alone:
`ΔS = 100·Δd_seg + 25·Δbytes/37,545,489`. A rung that costs 2,000 B and buys Δd_seg −2e-5 is
ΔS = −0.0013 + 0.00133 ≈ 0 — the arithmetic decides, not the direction.

## OPTIMAL FORM
- REFERENCE: the shipped semantic checkpoint at its shipped settings IS the reference. Pin its path
  + sha. Every rung is a matched delta.
- SCOPE reductions ALLOWED: a frame subset for early screening — but any rung that produces a
  VERDICT must be n600 (m88/m96: a prefix of a skewed population is a different population; use
  stratified-random n≥120 if you must subsample, NEVER a prefix).
- MECHANISM reductions FORBIDDEN: no d_seg from a proxy; no bytes from a projection where a real
  pack is available; no verdict from a re-quantization that was never actually decoded.
- Label: local Metal/CPU results are `[macOS-CPU advisory]`, `score_claim=false`.

## DELIVERABLE
The (bytes, d_seg, ΔS) table + a verdict on whether PR130's semantic operating point is optimal. If
it IS optimal, that is a strong finding — it says the distortion axis is closed on this base and the
whole remaining gap is rate. Report that plainly if it is what you measure.
