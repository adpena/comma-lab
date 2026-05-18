# Codex findings: G1 CPU-axis rerank + F1 receiver-path reframing

Date: 2026-05-18T21:46:50Z
Agent: Codex
Session: 019de465

## Finding 1: G1 existing-anchor rerank is stable, not a new score claim

The G1 CPU-axis rerank now runs through `tac.frontier_scan` and the literal
probe CLI `tools/probe_g1_cpu_axis_re_rank.py`. The live report is:

- `experiments/results/g1_cpu_axis_re_rank_20260518T214250Z/report.json`

Result:

- verdict: `FRONTIER_STABLE_VIA_RE_RANK`
- current CPU frontier: `0.1920513168811056`
- best existing CPU-axis anchor: PR101/fec6,
  `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- delta vs current CPU frontier: `+0.0000000000`
- canonical anchors scanned: 194
- qualifying CPU anchors scanned: 55
- no new score claim: `score_claim_valid=false`,
  `score_claim_kind=existing_anchor_rerank_no_new_score_claim`
- metadata bucketing: token-based metadata match only
  (`metadata_token_match_no_archive_sha_guess`), after Huygens caught the
  substring trap where `pr106_component_prefix16_pr101grammar` was being
  mis-bucketed as PR101

Interpretation: this closes the zero-cost existing-anchor rerank probe. It
does not retire CPU-axis-specific optimization. Future G1 movement depends on
new paired Linux x86_64 CPU anchors or a new archive family whose CPU axis beats
PR101/fec6.

## Finding 2: F1 internal-dim framing was a real conflation

Gauss found the important failure mode: PoseNet dims 7-12 are source-verified
unscored, but they are not archive-visible bytes. Mutating those tensors inside
a custom scorer harness proves only an internal algebraic invariant. It does not
prove contest-deliverable rate capacity.

Operator clarification fixes the canonical F1 shape:

- False framing: "dims 7-12 are a free byte channel in the archive."
- Correct framing: F1 is a scorer-blind RGB perturbation manifold, i.e. A2
  adversarial steganography specialized to PoseNet first-6-dim invariance.
- Receiver path: standard inflate-to-RGB remains the receiver path; no PoseNet
  or SegNet load at inflate time.
- Legal boundary: naive full-scorer/full-PoseNet extraction at inflate time
  remains rejected, but A1-SPECIALIZED is live under CLAUDE.md deterministic
  packet compiler / `contest_one_video_replay` when implemented as a tiny
  self-contained per-pattern byte transducer, fixed table, symbolic formula, or
  generated native code path and validated by exact CUDA auth eval.

Next F1 probe should measure controllable RGB perturbation capacity under
PoseNet first-6-dim stability and SegNet argmax stability, not mutate ephemeral
PoseNet output tensors.

## Finding 3: A1-SPECIALIZED is a canonical deterministic-packet path

The operator correction reopens A1 at the right granularity. My rejected variant
was the naive one: ship or load a full scorer / full PoseNet-like generic
inverter. The live variant is narrower and stronger: train or derive a
specialized inverter for the scored video and the chosen encoded patterns, then
lower it into the deterministic packet compiler as generated code, fixed tables,
distilled sparse/quantized weights, or a tiny native binary.

Required gate for A1-SPECIALIZED:

- self-contained packet; no hidden sidecars or network state
- all bytes charged to the archive unless fixed contest runtime already owns
  them
- typed runtime-consumption proof through `tac.packet_compiler`
- no scorer modification and no uncharged full scorer load at inflate time
- exact CUDA auth eval validates the archive/runtime pair before any score claim

This is not a contradiction with the F1 correction. F1 direct "dims 7-12 are
archive bytes" was a category error. A1-SPECIALIZED is a deterministic compiler
path for structural information not shipped as raw RGB payload.

## Verification

- `.venv/bin/python tools/probe_g1_cpu_axis_re_rank.py --write-report --json`
- `.venv/bin/python tools/cpu_axis_optimal_archive_selector.py --json`
- `.venv/bin/python tools/scan_best_anchor_per_axis.py --format text`

The initial `reports/latest.md` G1 wording exposed a scanner bug in my own
text: putting `+0.0000000000` next to `[contest-CPU]` caused the citation parser
to read the delta as a frontier score. The wording was corrected to "CPU-axis
delta, not a frontier score citation."
