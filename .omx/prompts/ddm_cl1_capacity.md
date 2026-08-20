# ddm_cl1 — THE MODEL↔TOKEN CAPACITY LADDER: measure d(tokens)/d(model), the gap-sized unknown

The rr1 recall audit recorded this derivative as **ABSENT**. It is the only named path to the
remaining ~30 KB.

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

## WHY THIS, WHY NOW

clean60 (60 epochs of HPAC retraining at PR130's FIXED capacity, `--channels 64 --patch 64
--delta 2 --frame-dim 8`) **LOST** by +185 B apples-to-apples. That is evidence that **more training
at fixed capacity does not beat their model** — which makes CAPACITY the untested axis, not epochs.

The single suggestive datapoint (SCOPED — two differently-TRAINED models, not a capacity ladder):
+96 packed model bytes coincided with −1,967 ideal token bytes ⇒ exchange ≈ −20.5 token B per model
B, ~20× past the −1 break-even. **Treat that as a hypothesis to test, NOT as a measured slope.**

## THE TASK

Run a PREREGISTERED small ladder on the REAL trainer flags (grep `add_argument` in
`train_hpac_self_compress.py` / `codec_hpac_integer.py` — do not invent):
`--channels`, `--patch`, `--delta`, `--frame-dim`, and `--rate-lambda` if it is real.

Per rung report **exact PACKED joint bytes**, not bpp: pack the model through
`pack_hpac_self_compress.py` (it carries a fail-closed bit-exact logit round-trip) and price the
tokens through the model's own entropy AND, where affordable, a real encode. State clearly which
is which — the trainer's `bpp` is the CROSS-ENTROPY (IDEAL), not a coded stream; conflating them
is the exact error that produced a false win earlier today.

**Kill a rung** if extra model bytes exceed token savings, or if the changed architecture breaks the
runtime/header contract, or if decode exceeds the 30-min budget.

## OPTIMAL FORM
- REFERENCE: PR130's shipped config IS the reference point (15,164 packed model / 114,852 ideal
  tokens). Every rung is a delta against it, measured the same way.
- SCOPE reductions ALLOWED: fewer epochs per rung (declare it), fewer rungs.
- MECHANISM reductions FORBIDDEN: no bpp-only rungs presented as byte rows; no rung without a real
  pack; no capacity change without the runtime-config adaptation it implies.
- Local Metal only. One fire at a time. Resumable + per-stage checkpoints (P0).

## DELIVERABLE
The ladder table + the FITTED d(tokens)/d(model) with its uncertainty, + a verdict: does growing the
prior pay, and where is the knee. If the slope is flatter than −1 anywhere, say so — that closes the
family and is worth more than a small win.
