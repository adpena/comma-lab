# ddm_dt1 — Is the ANS win AFFORDABLE? Measure the decode wall-clock DELTA before retaining any payload

**Owner:** codex arm · **Base:** PR130 CPR1 · scorer-free · `[macOS-CPU advisory]` · `score_claim=false`

## WHY THIS FIRES BEFORE THE PAYLOAD RE-RUN

`ddm_rc1_receiver` (`5de03569ad`) built the selector-explicit receiver and named a gate nobody had
on the board: n2 ANS decode took **2.683 s**; a **linear** extrapolation to n600 is **~805 s
(13.4 min) of ANS decode alone**, rendering SEPARATE and on top, against the contest's **30-minute
TOTAL** budget (`upstream/README.md:114`). If that holds, the −2,120 B rate win (ΔS −0.00141) may
cost more wall-clock than the budget can afford — and the expensive payload-retention re-run
(~4.7 GB of tables, 681 s, SSD chunking + resume receipt) would be wasted work.

**So measure the cost before paying it.** This arm is cheap and it GATES
`NEXT_IF_RESUMED` rank-3 (the ANS encode/retain arm).

## THE QUANTITY THAT ACTUALLY MATTERS — do not measure the wrong one

The absolute ANS decode time is NOT the number. The range coder's decode is **already inside the
shipped 30-minute budget**. The binding quantity is the **DELTA**:

```
Δt_decode = t_decode(ANS, n600) − t_decode(range, n600)
```

measured on the SAME token stream, SAME model tables, SAME host, SAME process shape. Report
Δt against the measured headroom, not against 30 min in the abstract.

**Also measure the current PR130 decode wall-clock end-to-end** (decode + render, the real
`inflate.sh` path) so "headroom" is a measured number and not an assumption. If nobody has ever
timed the shipped decode on this host, that is itself a finding — say so.

## PROTOCOL

1. **≥3 points, not 2.** RC1's 805 s came from a single n2 point extrapolated linearly. Measure at
   n ∈ {2, 8, 32, 120, 600} (or as far as time allows) for BOTH coders and fit the actual scaling.
   Report the fit form and its residuals — if it is not linear, say what it is. A 2-point linear
   extrapolation is the thing this arm exists to replace.
2. **Same object.** Reuse the recorded encode argv from
   `/Volumes/VertigoDataTier/pact/ddm_pr130_encode_tokens_metal_20260809/run/launch_manifest.json`
   and the committed receiver at `src/tac/pr130_runtime/fx1_runtime_tree/receiver.py`. The range arm
   must reproduce **116,980 B** on n600 — that equality is the faithfulness control and it has
   already passed once; if it fails here, STOP and report, do not proceed.
3. **Decode only, then decode+render.** Separate the two so the ANS delta is not hidden inside
   rendering cost.
4. **Contest host, honestly labelled.** We measure on macOS-CPU. The contest runs Linux x86_64 /
   T4. Label the axis; do NOT claim contest wall-clock from a macOS number. What transfers is the
   RATIO (ANS/range), which is far more portable than the absolute — say so explicitly and give
   both.

## FORK (pre-registered, so the verdict cannot drift)

- **Δt small vs headroom** → the −2,120 B is affordable. FIRE the payload-retention re-run
  (rc1 NEXT_IF_RESUMED rank-3: SSD atomic int16 chunks, retained constriction-0.5.0 ANS words,
  resume receipt) and the ANS lever proceeds to archive assembly.
- **Δt large vs headroom** → the ANS lever is **rate-positive but wall-clock-negative**, which is a
  real and publishable finding: it would explain why PR130 shipped a range coder, and it retires a
  lever we would otherwise keep spending on. Record it with the measured curve, and re-scope the
  −2,120 B as UNSHIPPABLE-ON-THIS-BUDGET rather than banked.
- **Δt uncertain** (fit unstable / host-dependent) → say so, name the measurement that would settle
  it, and do NOT promote either way.

Either fork is a win. Do not prefer the affordable one.

## OPTIMAL FORM

Reference form: a decode-throughput benchmark with ≥3 scale points, per-coder isolation, and a
declared fit. Declared reductions: SCOPE only (fewer n points if wall-clock forces it — say which
you dropped and why). MECHANISM reductions are TOY-BRACKET and cannot produce a family verdict —
in particular, timing a decode that does not actually reconstruct all 393,216 tokens per frame is
not a decode measurement.

Provenance pins (verify, and STOP if any fails to reproduce):
- receiver `5de03569ad` (`src/tac/pr130_runtime/fx1_runtime_tree/receiver.py`)
- archive sha `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`, 191,052 B
- ANS length receipt `ans_n600/ans_vs_range_n600_result.json` (range 116,980 / ANS 114,860)
- constriction **0.5.0 pinned** — RC1's n2 exactness proof is pinned to it; record the version you run
- RC1 receipt `.omx/research/ddm_pr130_reproduce_20260809/DDM_RC1_RECEIVER_RECEIPT.md`

## HARD RULES

- Bulk artifacts → `/Volumes/VertigoDataTier/pact/ddm_dt1_20260809/`. No `/tmp` in persisted evidence.
- Commits via `tools/subagent_commit_serializer.py`, POST-EDIT `--expected-content-sha256`, tags
  `[no-triality] [p0-ledger-ok]`, **no attribution trailer of any kind**.
- `.py` files: 2 × `tools/review_tracker.py mark-file <f> --status reviewed`; never
  `REVIEW_GATE_OVERRIDE=1` with a `.py`.
- Intake clone at `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/` is READ-ONLY.
- Nothing here is a score. Label every number and state the host.

## DELIVERABLE

The measured Δt curve for both coders, the fit and its residuals, the ANS/range RATIO, the measured
end-to-end PR130 decode wall-clock on this host, the fork verdict with its scope, and an explicit
list of rungs not run.
