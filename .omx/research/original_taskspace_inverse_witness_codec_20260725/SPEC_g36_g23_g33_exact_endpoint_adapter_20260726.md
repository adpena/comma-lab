# G36 specification — exact G23/G29 to G33 endpoint adapter

Date: 2026-07-26  
Lane: `lane_g36_g23_g33_exact_endpoint_adapter_20260726`  
Status: strict G29 preflight adapter-readiness auditor implemented; real G33 base/endpoint blocked on named missing receipts  
Authority: original Pact research-only composition; no score, candidate, promotion, dispatch, or pointer mutation

## Purpose

G36 is the typed seam between the selected-solution compiler, the public
workflow observation, and the exact receding-horizon controller. It does not
copy fields between receipts. It reopens every parent, proves that all parents
describe the same physical archive and public output, and constructs one G33
base or endpoint without laundering research evidence into contest authority.

The adapter consumes:

1. one exact `G17WholeObjectStateV1` for counted representation, placement,
   receiver lifecycle, and exact whole-object interaction custody;
2. one G29 `PublicEvaluatorExecutionReceiptV1` with its embedded clean-run,
   mirror-observation, input-ledger, output-ledger, equality, trace, ABI, and
   public-workflow parents;
3. one exact continuation/action-space receipt;
4. the current content-addressed dynamic frontier pointer; and
5. for action endpoints, an exact G17 effect/corner plus its G33 family, scale,
   base identity, and epoch.

G29 currently and correctly remains `research_only=true`. Therefore G36 may
construct `RESEARCH_ADVISORY` states only. Production contest-CPU/CUDA modes
remain impossible until a separate governed execution boundary exists; an
axis label cannot upgrade authority.

## Five identity domains

These domains are deliberately non-aliasing:

1. **Counted representation identity** — exact archive bytes/SHA/length,
   member map, physical coding groups, decoder program/operands, and placement.
2. **Present evaluator-cell semantic identity** — a domain-separated hash of
   only the complete ordered sequence
   `(pair_index, candidate Seg argmax cell, candidate Pose6 fp32 cell)`.
3. **Score sufficient statistics** — ordered Seg mismatch counts and Pose MSE
   fp32 values plus exact aggregates and final complete-ZIP rate.
4. **Continuation/action-space identity** — every future-relevant degree of
   freedom and hard constraint needed to determine which actions remain
   reachable after this state.
5. **Proof-dependency identity** — archive/output/input/target/scorer/runtime/
   ABI/axis/trace/mirror/equality receipts that certify the other domains.

The semantic evaluator-cell identity MUST NOT contain archive identity, raw
video identity, hardware axis, target cells, scorer-input identity, upstream
snapshot, static authority manifest, proof receipt, or continuation state.
Those belong to domains 1, 4, or 5. A context-rich cell digest may be retained
as proof identity, but it cannot substitute for present semantic state.

Target Seg/Pose cells are encoder/evaluator evidence only. They never enter the
candidate archive, generic decoder code, continuation state, or semantic
candidate-cell identity.

## Exact same-object joins

Construction refuses unless all applicable joins close:

- G17 archive bytes, SHA, length, member, placement manifest, and counted
  physical groups match the G29 compiled/public archive exactly;
- G29 run A, run B, equality, and execution agree on archive, public raw output,
  complete ordered input ledger, complete ordered output ledger, ABI, axis,
  score components, report, and public entrypoint;
- both observation mirrors are explicitly non-authoritative and each binds the
  exact clean official run inputs/report/process trace;
- the output ledger recomputes aggregate `d_seg` and aggregate `d_pose` exactly
  as retained by both clean runs and the public execution receipt;
- the final score recomputes through `tac.contest_score` from those aggregates
  and the exact complete archive bytes;
- the public output is exactly 1,200 frames with the frozen geometry/order;
- all video-specific state is counted and every generic decoder dependency is
  inside the reviewed recursive runtime closure.

G17's private realized-pair/chronology bytes are not assumed to be the public
RGB raw video. If they are different representations, a typed receiver/raster
bridge receipt must bind their exact transformation to G29's public raw bytes.
Comparing unrelated hashes or silently treating a private intermediate as the
public output is forbidden.

## Continuation equivalence

The continuation receipt conservatively binds:

- decoder ABI and generic instruction semantics;
- factor graph, dictionary/context state, shared reservoirs, and ownership;
- counted payload placement and remaining movable/factorable groups;
- reachable typed action generators at every required family and scale;
- topology/gauge/transport/repair/training state that changes future actions;
- public workflow time, memory, determinism, portability, and access limits.

It does not hash the endpoint as a shortcut. Equal present cells with different
reachable actions remain different states; equal cells and equal continuation
may quotient to the cheapest counted representation.

## Interaction preservation

Every complete G17 counterfactual corner lattice is reopened and bound into an
exact interaction-hypergraph receipt. Partial corners are proposal evidence
only. Exact sign reversals are preserved, including locally harmful actions
that become beneficial in a measured composition. G7 and Consumer-15 outputs
may order acquisition, but cannot filter the endpoint universe or certify
family-by-scale coverage.

Higher-order terms are not approximated by pairwise sums. Complete rebuilt
whole-object endpoints absorb all orders. An unmeasured terminal subtree may
enter the exact manifest partition only through G38 proof bytes that are
reopened with their independent context, exact generator-manifest payload,
typed measurement/constraint dependencies, and exact upstream score order.
Family-by-scale occupancy is audit telemetry and cannot certify closure.

## G33 construction

For a base, G36 derives every `WholeObjectBaseV1` field, including passing
`PUBLIC_EVALUATE_SH_WHOLE_WORKFLOW` constraints, from reopened parents. For an
endpoint it additionally proves exact base/epoch custody, action family/scale,
proposal source, interaction graph, and complete public workflow evidence.

`measurement_receipt_sha256` identifies exact score-sufficient-statistic
custody. `evaluator_output_cell_ledger_sha256` identifies the complete proof
ledger. `evaluator_cell_identity_sha256` is the pure present semantic sequence.
`continuation_equivalence_sha256` and its proof receipt remain separate.

No G36 object authorizes production commit, paired-axis promotion, or pointer
movement. A target crossing remains a research observation until the same
archive receives separately governed contest-CPU and contest-CUDA custody.

## Acceptance tests

Focused tests must prove:

1. exact valid research-only base and endpoint construction;
2. archive/raw/axis/ABI/input/output/score/placement mismatch refusal;
3. caller-authored receipt bytes cannot upgrade research authority;
4. equal candidate cells across different archives share semantic identity
   while retaining different representation/proof identities;
5. equal present cells with different continuation receipts do not quotient;
6. target-cell or context inclusion in semantic identity is rejected;
7. private-intermediate/public-output hash substitution is rejected;
8. incomplete output ledgers, mirror equivalence, corner lattices, exact
   generator-leaf partitions, or proof-byte bundles block construction;
9. total public workflow, not inflate-only time, controls the hard limit; and
10. parse/re-emit, caller-order invariance, Ruff, and deterministic focused
    tests pass.

## Current blocker and pointer honesty

G29 is test-stable for compile, placement, decoder ABI, generic-source audit,
recursive dependency discovery, and execution-readiness receipts. G36 now
strictly reopens and joins those seven roles in
`taskspace_g23_g29_g33_endpoint_adapter.py`; the CLI
`tools/audit_taskspace_g23_g29_g33_endpoint_adapter.py` materialized
`g36_endpoint_adapter_readiness_g29_preflight_20260726.json` from the retained
SSD receipts. The exact joined object is the 80,238-byte archive
`68351f57781d8fe60c05ab59fc250e48d6bb03e7cdf95b3d00987328d08d2a98`.

Against the content-addressed dynamic pointer observed by that receipt, its
exact rate term is `0.053427190680616785`; the remaining *coupled distortion
score budget* is `0.1185728093193832`. The zero-Pose and zero-Seg intercepts,
which are geometry diagnostics rather than independent acceptance thresholds,
are respectively `d_seg=0.001185728093193832` and
`d_pose=0.001405951110989081`.

The receipt fails closed on six exact roles: a strict reopenable G17
whole-object state; a typed G17-private-to-G29-public raster bridge; a real G29
public evaluator execution; the exact G33 generator-domain manifest; the G33
action-continuation receipt; and a durable G38 proof-byte bundle. In particular,
`G17WholeObjectStateV1` remains an in-memory object graph rather than a strict
receipt, and G17 chronology bytes cannot be assumed equal to G29's 1,200-frame
public RGB raw output. G36 therefore cannot materialize a real base or endpoint
yet and cannot launder G29's structurally valid but Pose-catastrophic represented
solution into evidence for the low-distortion lattice teacher.

Pointer moved: **no**.  
Frontier score claim: **no**.  
Candidate archive: **no**.  
Dispatch: **none**.

## Stores consulted

- `CLAUDE.md`, `AGENTS.md`, live lane/subagent state, and current pointer;
- G17/G23 selected-solution compiler and effect-hypergraph surfaces;
- G29/G31 public recursive closure source and audit findings;
- G33 exact receding-horizon controller/spec;
- real G14 interaction receipt containing 12 exact sign reversals; and
- G35 bounded-R10 receipt and remaining action-universe blockers.
