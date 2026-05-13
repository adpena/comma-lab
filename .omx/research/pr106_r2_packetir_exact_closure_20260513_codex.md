# PR106/R2 PacketIR Exact-Eval Closure - 2026-05-13

Scope: close the PR106/R2 PR101-grammar plus low-level Brotli repack line without
promoting it past the current exact-CUDA frontier or re-dispatching the same SHA.

Concrete artifact:

- Closure JSON: `experiments/results/pr106_r2_packetir_exact_closure_20260513_codex/closure.json`
- Closure note: `experiments/results/pr106_r2_packetir_exact_closure_20260513_codex/closure.md`
- Tool: `tools/build_pr106_r2_packetir_exact_closure.py`
- Reusable module: `src/tac/packetir_exact_closure.py`

Verdict:

- classification: `exact_measured_not_current_frontier`
- candidate SHA: `287e6edc612803a9a9d5de3ce50b421c039704f38bae442a6dcc97a3e8d6ed4d`
- candidate bytes: `186629`
- source SHA: `c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383`
- byte delta versus PR101-grammar source: `-151`
- [contest-CUDA] score: `0.2065174760196528`
- [contest-CPU] score: `0.22796397327358284`
- delta versus PR101-grammar [contest-CUDA] source: `-0.00010065943776227382`
- delta versus HLM1 [contest-CUDA] current-best reference: `+0.0001371669443431811`

Dispatch decision:

Do not dispatch this same candidate archive again. The candidate is byte-closed,
axis-labelled, and already exact-evaluated; it improves its PacketIR source line
but does not beat the current HLM1 [contest-CUDA] reference. Reactivate only for
a new runtime-consumed PacketIR/low-level candidate with a new archive SHA and a
path to exact-CUDA score below the current best.
