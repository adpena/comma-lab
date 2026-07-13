# DIRECTIVE → ane_unlock_correction (live arm) — fold the ecosystem survey (main, 2026-07-13)
Per the directive protocol this supersedes/extends your prompt. The ecosystem survey landed
(.omx/research/ane_ecosystem_survey_20260713.md — 17 techniques with receipts). Fold into your ladder:
1. R1 EXACT API: use `ct.transform.FP16ComputePrecision(op_selector)` — per-op fp32 for the logit head +
   R0-identified offender ops INSIDE one model (no surgery/pipeline split needed; community analogue cost ~3%
   latency vs 3× full-fp32). This is your R1's cheapest realization — try it FIRST.
2. R0 ROUTING DATUM (main-local, committed 56a4b213a1): MLX full-forward fp16 flip rate = 0.0584% vs ANE 2.47%
   ⟹ fp16 arithmetic is benign; the ANE excess is MIL op-substitution/ANE-specific — aim R0 at op-level diffs
   (resize/pool/SE candidates), and apply survey item 4 (Anemll fp16-preflight weight PRE-SCALING on B2's
   SE-pool/BN-fold hot spots — free, offline, math-preserving).
3. ACCEPTANCE METRIC UPGRADE (survey item 3, WhisperKit QoI): per-pair flip-rate vs fp32, WORST-PAIR not mean,
   as the gate for every rung (your preregistered aggregate bar stays; add the worst-pair column).
4. R4 REFRAME (survey item 5): the dependable M5 W8A8 win is SRAM-FIT — B2 fp16 weights ~31MB brush the
   measured 32MB ANE SRAM cliff (−30% beyond it); int8 COMPUTE speedup is contested between sources — A/B it,
   don't assume.
5. RESIDENCY: CPU_AND_NE is a REQUEST not a guarantee — add a residency check (powermetrics ANE counters or
   Xcode perf report equivalent) to every latency receipt so a silent CPU-fallback never masquerades as an
   ANE number.
6. NEW RUNG R6 (throughput, after fidelity): Anemll-style enumerated batch-tier multifunction compile
   (fwd_b1/b8) + weight dedup (~+55% batch gain, arXiv:2606.22283) — receipt-level design only unless cheap.
CONCURRENCY: qualified GREEN derived; the $0 solo-vs-concurrent self-measurement (ANE forward ∥ MLX step +
powermetrics, accept <5% mutual degradation) is assigned to MAIN, not you.
