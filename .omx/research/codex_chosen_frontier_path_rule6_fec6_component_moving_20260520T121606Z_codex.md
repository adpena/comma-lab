# Codex Chosen Frontier Path - Rule #6 / FEC6 Component-Moving Packet

**UTC:** 2026-05-20T12:16:06Z
**Decision owner:** Codex
**Chosen lane:** `lane_pr101_fec6_packetir_compiler_identity_queue_20260519`
**Score claim:** false
**Promotion eligible:** false
**Dispatch authorized:** false

## Decision

Choose the Rule #6 / FEC6 component-moving packet path as the next
frontier-moving artifact path.

The immediate artifact is not another memo and not a paid dispatch. It is a
byte-closed local candidate archive, produced from the existing PR101/FEC6
PacketIR surface, where a grammar-level selector or procedural-residual change
is consumed by the runtime and can plausibly move Seg/Pose components. Only
after local consumed-packet proof exists should paired CPU/CUDA exact eval be
considered.

## Why this path

Current frontier state is split by axis:

- `[contest-CPU]`: `0.1920513169`, archive
  `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`,
  PR101/FEC6.
- `[contest-CUDA T4]`: `0.2053300290`, archive
  `9cb989cef519ed1771f6c9dc18c988ee93d01a2925da1913d63f9015d6247cf4`,
  PR106 format0d.

Same-runtime FEC6 byte-only polish is below the useful threshold: the selector
entropy profile shows roughly 2-8 bytes of plausible selector-header savings,
while strict `<0.192` on the current CPU anchor needs about 78 charged bytes if
components are unchanged. Therefore the next FEC6 action must move components
or add a consumed score-affecting packet layer.

The VQ K=2 diagnostic is terminalized as diagnostic-only and should not receive
more paid fan-out without a K-dependent archive grammar. The TT5L doctor path
remains valid, but it starts with provider/source-manifest hygiene; the FEC6
PacketIR path is closer to a local artifact because identity/runtime-consumption
proofs already exist.

## Starting artifacts

- Candidate queue:
  `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/packetir_candidate_queue.json`
- Candidate queue memo:
  `.omx/research/pr101_fec6_packetir_candidate_queue_20260519_codex.md`
- Runtime consumption proof:
  `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/runtime_consumption_proof_20260520_codex.json`
- Mirrored runtime proof:
  `.omx/research/pr101_fec6_runtime_consumption_proof_20260520T065500Z_codex.json`
- Operator-space manifest:
  `experiments/results/fec6_selector_operator_space_20260517_codex/operator_space_manifest.json`
- Authority matrix:
  `.omx/research/pr101_fec6_frontier_packetir_matrix_20260519_codex.md`

Observed queue state:

- `33` candidates total.
- `29` operator candidates.
- `0` newly materialized candidate archives.
- All non-identity candidates remain blocked by materialization, runtime
  consumption/no-op proof, and paired exact-eval custody.

## First artifact to build

Output directory:

```text
experiments/results/pr101_fec6_rule6_component_candidate_20260520_codex/
```

Required first outputs:

1. A materialized `archive.zip` candidate whose member bytes differ from the
   current FEC6 archive.
2. A manifest with archive bytes, archive SHA-256, member SHA-256, changed
   PacketIR section(s), and selected operator rationale.
3. A runtime-consumption/no-op proof showing the changed bytes are consumed by
   `submission_dir/inflate.py` and the PR101/FEC6 codec path.
4. A local inflate smoke or exact failure classification.
5. An explicit `score_claim=false`, `promotion_eligible=false`, and
   `ready_for_exact_eval_dispatch=false` status unless and until paired
   contest CPU/CUDA exact eval is run on the same archive/runtime.

## Candidate selection rule

Prefer a grammar-aware selector or procedural-residual operator that:

- changes decoded selector/runtime state, not ZIP wrapper bytes only;
- has a plausible component-moving rationale from pair/component rows or a
  deterministic perturbation test;
- has nonpositive or bounded byte delta after archive materialization; and
- survives runtime parsing without scorer imports or network dependencies.

If no row satisfies this, terminalize the artifact as
`exact_failure_classification=packetir_queue_has_no_materializable_component_candidate`
and then move to the TT5L doctor/manifests path rather than spending on weak
FEC6 byte polish.

## Gating

Do not dispatch from this memo. Dispatch requires a new lane claim, the
materialized candidate archive, local consumed-packet proof, archive/runtime
custody, and paired-axis plan. Any score language before that is forbidden.
