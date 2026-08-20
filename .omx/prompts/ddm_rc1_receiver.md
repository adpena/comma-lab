# ddm_rc1 — BUILD OUR RECEIVER: the 3-stream model parser + ANS LIFO token decoder

**You are the critical path.** 2,983 measured lossless bytes (ΔS −0.0019864, S 0.172141297 →
≈0.170154897) are banked and UNSHIPPABLE because no receiver parses the new formats. Your job is to
make them shippable.

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

## THE TASK

Build, in OUR repo (not the read-only intake), a receiver that decodes BOTH new formats:

**Leg A — 3-stream model parser.** Format is
`[u32 sem_len][u32 car_len][u32 hpac_len][brotli(sem)][brotli(car)][brotli(hpac)]`
replacing the single `LZMA(models_raw)`. Must reconstruct `models_raw` BYTE-IDENTICALLY
(sha match is the gate) and be consumed by the real model loaders.

**Leg B — ANS LIFO token decoder.** `constriction.stream.stack.AnsCoder`. The encoder must
`encode_reverse` over the sequence so the decoder POPS FORWARD, preserving causal AR context.
Encode-side needs the conditional tables materialized (117.9M × 5 ≈ 2.4 GB fp32) where the current
encoder streams them — a co-running n600 job is measuring exactly this; read
`/Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/ans_vs_range_n600.log` before designing
the memory path, and chunk if it OOM'd.

**The gate that matters:** an end-to-end round-trip on the REAL archive — encode → decode → the
decoded token field is bit-identical to the input token field, AND `models_raw` sha matches. Both
legs, executed, not argued.

## OPTIMAL FORM

- REFERENCE form: PR130's own `inflate.py` + `codec_hpac_integer.decode` (read-only intake,
  pinned `codec_hpac_integer.py@` current sha — record it). Our receiver must be a strict
  functional superset for these two formats.
- SCOPE reductions ALLOWED: fewer frames for the round-trip smoke (then scale to n600).
- MECHANISM reductions FORBIDDEN without an explicit TOY-BRACKET declaration: no "assume brotli
  present" without the ImportError fallback path decided; no "decode only the first group"; no
  skipping the causal-context reconstruction.
- rule-118: decoder code is GENERIC ALGORITHM = FREE, zero counted bytes. Do not smuggle any
  video-derived table into it.

## DELIVERABLE
A committed module + tests + a receipt naming: both round-trips executed w/ sha equality, the
measured decode wall-clock vs the 30-min budget, and the brotli-dependency decision (REQUIRED
self-install for the −903 B leg vs the dep-free lzma2 variant at −234 B — state the trade, do not
silently pick). If a leg fails, say which and why; a half-built receiver honestly reported beats a
claimed one.
