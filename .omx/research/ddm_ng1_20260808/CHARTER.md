# ddm_ng1 — modded-nanogpt PR #349 (ANVIL) crosswalk vs the wall-clock/optimizer/methodology surfaces (operator drop 2026-08-08)

Tags: [no-triality] [p0-ledger-ok]. CODEX ARM, rigor-triage-first (the if1/fa1/lw1 protocol).
Read `docs/operating_manual_craft_handoff.md` + CLAUDE.md/AGENTS.md first. RECALL-FIRST (m44):
consult receipts, never working memory — #685 px1 (SOAP/Muon crosswalk), #849 (Tilde/KL-Shampoo),
#556 (FilmPolarSPD queued), dy2 (plateau-anchored tail-average EMA — BUILT), #470/ffm1 (int8/fp
rungs), wc3 f9ab8fb399 + WC3_FINDINGS.md (renderer-bwd 57%, saturated 1.361×, F1 two-point
schedule support), gc21 GC21_CONVOCATION.md (event predicate + fp16 2.0e-6 guard + "MLX
schedule/noise envelope" — currently UNCALIBRATED), #414 (KellerJordan GitHub swept PR95-era —
STALE, speedrun history since unswept).

## The drop

https://github.com/KellerJordan/modded-nanogpt/pull/349 — "New record: 1.082 min (64.95s) —
new ANVIL optimizer + full-stack fp8 and schedule overhaul" (Deven Pietrzak). −12.7% vs PR #89's
74.38s, same-session baseline retime, validation CE 3.27604 (σ 0.0009, n=16 unseeded, p<1e-10 vs
the 3.28 gate). Mechanism per PR text: whitened velocity (six-map spectral whitening cascade,
composite envelope [0.9971,1.0095]) over twin fast/slow momentum rails w/ Nesterov lookahead +
per-lane energy equalization at fixed Frobenius norm + sign-aligned decay + tail-blend weight
averaging folded into the shipped model; full-fp8 MLP fwd/bwd; narrow Q/K (8/10 layers head-width
64); 40 mid-schedule grown steps; period-4 accumulation cadence on embedding Adam. Review thread:
jvarho caught timing-critical ops moved OUTSIDE the timed region; author retimed with them inside
— record held at 64.95s.

## RIGOR TRIAGE FIRST (before any adoption row)

Verify at source (fetch the PR diff + thread, not the summary): (a) the ANVIL mechanism as coded
vs as described (is the "six-map cascade" tuned Newton-Schulz? extract the actual coefficients);
(b) the timing-region resolution actually landed (post-fix numbers); (c) hardware/protocol (the
speedrun target + hardware class — do NOT trust the intermediate summary's CPU claim); (d) is the
PR MERGED or pending review — a pending record is a CLAIM, label accordingly.

## Crosswalk questions (each → ADOPT / ADOPT_CLASS / RACE / LESSON-ONLY / N-A, w/ named consumer + falsifier)

1. **σ-calibration methodology → M1 ticket (RANK-1 CANDIDATE, cheapest + fire-gate-relevant).**
   Their n=16 unseeded runs → σ → significance gate vs target is the calibration our M1 event
   predicate + fp16 2.0e-6 guard REFERENCE but do not HAVE (F1: two-point support, Δ3.2e-7).
   Candidate ADOPT: N repeated short same-config runs (bench harness exists, ~2 min each at 5
   steps) → measured σ of the fp32 GPU sanity metric → the envelope constant in the gc21 STOP
   rule + the fp16 guard becomes MEASURED not assumed. Consumer: M1 seal (ticket field
   `sanity_sigma_measured`). Falsifier: σ ≥ the fp16 guard bar (2.0e-6) ⇒ the guard is
   undiscriminating and must be re-derived.
2. **Timed-region boundary discipline → bench/profiler CLASS rule.** Their reviewer-catch =
   our fp16 instrument-scope flag, same genus. Candidate ADOPT_CLASS: a one-line timed-region
   manifest in bench receipts (what is inside/outside the wall-clock) so region drift is
   detectable. Consumer: wc1 bench (rides the owed append-mode fix).
3. **ANVIL vs our Muon finishing stage → RACE-not-reputation.** Extract exact update rule +
   coefficients; is dual-rail momentum + tuned-envelope whitening portable to the TR1/receiver
   trainer (MLX AdamW today at M1; Muon = witness finisher)? Compose w/ #685's update-RMS-matching
   fairness methodology for any A/B. Consumer: post-M1 optimizer window; #556 queue re-rank.
4. **Tail-blend weight-averaging fold → dy2 convergent evidence.** Their shipped-model tail
   average vs dy2's plateau-anchored tail-average EMA mode (BUILT, unfired A/B). Does their form
   (window, weighting) differ from dy2's? Consumer: dy2's A/B arm design.
5. **fp8 fwd/bwd → precision-ladder rung below our fp16.** MLX fp8 support status (verify — mx
   dtype coverage); if unavailable, LESSON-ONLY with the trigger condition named. Consumer:
   saturated-config successor.
6. **Mid-schedule growth (40 grown steps) + accumulation cadence → #686 event-schedule doctrine.**
   Grown-batch/window mid-run vs our event-driven continuation — complementary or contradicting?
   Consumer: M1 EXTEND_WITH_RESUME branch.
7. **Speedrun leaderboard sweep (STALE #414 refresh).** The record HISTORY since PR95-era sweep:
   enumerate record-to-record deltas + their mechanisms → which classes are未 raced on our
   vehicle. Consumer: wall-clock lever table (wc3 findings §6).
8. **Narrow Q/K + bigram sink** — expected N-A (transformer-specific); confirm honestly, no
   forced transfer.

## OPTIMAL FORM

Reference form = the standing drop-crosswalk family (if1/fa1/lw1/gdl1). No mechanism reduction:
source-verified claims only (PR diff + thread, not summaries), every row typed w/ falsifier +
named consumer, rigor-triage section FIRST, honest N-A rows. SCOPE bounds: $0, no Metal/scorer
runs, no launch; bench σ-runs (Q1) are DESIGN-only here — MAIN fires them on the ticket path.
Provenance pins: wc3 @ f9ab8fb399 · gc21 @ the spawning commit · dy2/px1/tilde receipts by path.

## Deliverables

`.omx/research/ddm_ng1_20260808/NG1_CROSSWALK.md` — triage results + the 8 typed rows + ranked
op-routables. Commit via `tools/subagent_commit_serializer.py` w/ post-edit shas, tags
[no-triality] [p0-ledger-ok]. Boundaries: research-only; the M1 σ-calibration (Q1) returns as a
DESIGN MAIN executes at ticket-seal; no run mutation; frontier line = own-vehicle per the
end-of-turn law.
