# Omega-OPT Anchor Discipline Ledger - 2026-05-08

Scope: adversarial review of the nested Omega-OPT score claims. This ledger
preserves the research designs while blocking score promotion, ranking, or
method retirement until a matching 1:1 archive/eval anchor exists.

Core rule: every score number below is a hypothesis unless the exact archive
bytes are consumed by `inflate.sh` and evaluated through
`upstream/evaluate.py` on CUDA with full component custody. CPU, MPS, proxy,
post-hoc byte measurements, synthetic convergence, and design manifests can
route the next build, but they cannot promote, rank, or retire the method.

Fail-closed fields for every unanchored row:
`score_claim=false`; `promotion_eligible=false`;
`rank_or_kill_eligible=false`; `ready_for_exact_eval_dispatch=false`.

## Claim Classification

| claim_id | Predicted score | Current classification | Current anchor status | Missing anchors | Next 1:1 test | Fail-closed fields |
|---|---:|---|---|---|---|---|
| omega_opt_linear_stack | 0.130 | prediction | CPU byte anchor only; no score anchor | 1:1 composition config, runtime packet consumption, full retrain or faithful no-retrain disclosure, exact CUDA auth eval | Build one PR101 stack archive from the declared arch_shrink, IMP, lossy coarsening, and brotli sequence; record old/new archive SHA-256; run exact CUDA auth eval | score_claim=false; promotion_eligible=false; rank_or_kill_eligible=false; ready_for_exact_eval_dispatch=false |
| omega_opt_multipass_imp_cycle | 0.115 | prediction | no matching config anchor | Q-FAITHFUL five-stage config, IMP-cycle archive materialization, per-pass charged-byte ledger, exact CUDA auth eval | Freeze a minimal Q-FAITHFUL plus one IMP-cycle config, emit one final contest packet, and compare exact CUDA auth eval against the non-IMP anchor | score_claim=false; promotion_eligible=false; rank_or_kill_eligible=false; ready_for_exact_eval_dispatch=false |
| omega_opt_hstack_of_vstacks | 0.110 | design | never built | parser-proven stream map, serial VStack manifests, parallel HStack merge manifest, runtime packet closure, exact CUDA auth eval | Use the codec-stack planner to materialize one parser-proven component map on PR101 or PR106, then build an identity packet before changing bytes | score_claim=false; promotion_eligible=false; rank_or_kill_eligible=false; ready_for_exact_eval_dispatch=false |
| omega_opt_joint_admm_cross_component | 0.105 | prediction | engine exists; no component allocation anchor | CodecOp-to-proximal adapter, real component budget ledger, old/new packet SHA-256, exact CUDA auth eval | Run Joint-ADMM over one real PR101/PR106 component budget with fixed proximal codecs; emit the selected allocation packet and exact-eval it on CUDA | score_claim=false; promotion_eligible=false; rank_or_kill_eligible=false; ready_for_exact_eval_dispatch=false |
| omega_opt_bilevel_optimization | 0.100 | prediction | scaffold only; no convergence or archive anchor | outer convergence proof, meta-Lagrangian selection artifact, inner ADMM allocation artifact, byte-closed archive, exact CUDA auth eval | Run a one-phase bilevel slice on a fixed substrate with a locked atom ledger; emit one archive and require convergence diagnostics plus exact CUDA auth eval | score_claim=false; promotion_eligible=false; rank_or_kill_eligible=false; ready_for_exact_eval_dispatch=false |
| omega_opt_score_feedback_meta_pass | 0.095 | prediction | never run | score-feedback loop log, dispatch-claim custody for every eval, archive identity chain, terminal exact CUDA auth eval | Run two score-feedback meta-passes on the same archive family; record archive SHA-256 transitions; treat only the terminal exact CUDA eval as score evidence | score_claim=false; promotion_eligible=false; rank_or_kill_eligible=false; ready_for_exact_eval_dispatch=false |
| omega_opt_per_tensor_hstack_of_vstacks | 0.092 | design | design only | per-tensor stream grammar, per-tensor HStack grouping manifest, cross-language conformance vectors, runtime packet, exact CUDA auth eval | Select two tensors, emit canonical byte vectors for one serial transform and one parallel grouping, then build a minimal runtime packet that consumes those sections | score_claim=false; promotion_eligible=false; rank_or_kill_eligible=false; ready_for_exact_eval_dispatch=false |
| omega_opt_recursive_residual_gradients | 0.090 | prediction | aspiration only | implemented transform, stable convergence or fail-closed divergence criterion, charged-byte residual packet, exact CUDA auth eval | Prototype one residual-gradient correction stage with fixed iteration count and deterministic payload accounting; compare exact CUDA auth eval against the unchanged anchor | score_claim=false; promotion_eligible=false; rank_or_kill_eligible=false; ready_for_exact_eval_dispatch=false |

## Reactivation Criteria

Reactivation means "eligible for the next exact-evaluable prototype," not
promotion. A row becomes promotion-eligible only after the prototype produces:

- exact archive bytes and archive SHA-256 before and after the transform;
- score-affecting payload proof showing `inflate.sh` consumes the changed bytes;
- no sidecars, network, scorer load, or local-state dependence at inflate time;
- full-sample `contest_auth_eval.json` from the canonical CUDA path;
- component distances and recomputed score tied to the exact archive SHA-256;
- terminal dispatch-claim row when any remote GPU eval was used.

## Tooling Contract

The canonical machine contract is `src/tac/omega_opt_claims.py`.
`tools/check_omega_opt_anchor_discipline.py --strict` validates this ledger,
Omega-OPT evidence rows, lane-registry promotion, and optional generated
planner manifests. The HStack/VStack planner now embeds the same eight-row
claim table in its nested optimization metadata so generated plans remain
design artifacts until the exact-anchor fields exist.

## Adversarial Notes

- The existing linear-stack CPU byte measurement is valuable as a byte anchor,
  but it does not anchor the predicted 0.130 score because the scored runtime
  packet and exact CUDA eval do not exist.
- The Joint-ADMM engine is real infrastructure, but the Omega-OPT 0.105 number
  specifically needs component allocation plus a byte-closed archive.
- The bilevel and score-feedback rows are not invalidated. They are kept as
  hypotheses with concrete build paths and fail-closed promotion semantics.
