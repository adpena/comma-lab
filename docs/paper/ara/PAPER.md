---
ara_version: 0.1
ara_paradigm: "Agent-Native Research Artifact (arXiv 2604.24658)"
title: "Task-Aware Video Compression: From Codec Post-Filtering to Neural Renderer"
project: comma-lab
project_local_alias: pact
project_track: "comma.ai video compression challenge"
deadline: 2026-05-03
last_updated: 2026-04-30
status: draft
audiences:
  - human_reviewer: "comma.ai writeup-prize judges"
  - agent_reviewer: "Ara-native review system (sufficiency check)"
  - human_skim: "challenge participants and self-driving researchers"
disclosure_policy:
  public:
    - "[contest-CUDA] tagged scores only"
    - "engineering rigor (78 STRICT preflight checks)"
    - "Era 1 + Era 2 narrative, anonymized leaderboard"
  private:
    - "Lane W (hard-pair self-compress) recipe"
    - "Lane Omega (Hessian-aware quantization) schedule"
    - "Lane DARTS-S architecture-search recipe"
    - "Specific Selfcomp paradigm sequencing decisions"
    - "Internal supplemental URLs and credentials (until operator approves a surface)"
score_lanes:
  contest_compliant_floor: 1.043987524793892
  contest_compliant_floor_evidence: "experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/eval/contest_auth_eval.json"
  contest_compliant_units: "[contest-CUDA A++]"
  unlimited_compute_used: false
  unlimited_compute_units: "[unlimited-compute]"
layer_index:
  logic: "logic/  — scientific reasoning (problem, claims, related work)"
  src: "src/    — physical layer (kernel index + repository pointer)"
  trace: "trace/  — exploration graph (decisions, dead-ends, pivots)"
  evidence: "evidence/ — raw outputs (results, logs, artifacts)"
---

# PAPER.md — Ara root manifest

> Historical draft artifact, not a current public disclosure surface. The
> frontmatter preserves the 2026-04-30 Ara packaging intent, including
> private-disclosure policy fields and stale score floors. Do not publish,
> cite, or compile this manifest into public docs until it has been refreshed
> and redacted against the current `README.md`,
> `docs/terminology_and_boundaries.md`, and evidence-grade score tables.

This is the agent-native research artifact for the comma.ai video compression
challenge entry produced by the `comma-lab` workspace. The local checkout was
historically named `pact`; use `comma-lab` for public/release references. It follows the Ara paradigm
(Agent-Native Research Artifacts, arXiv 2604.24658): a four-layer
machine-executable knowledge package that replaces a single PDF narrative.

## Triage (read this in 30 seconds)

- **What we did:** task-aware video compression that backpropagates through
  frozen PoseNet and SegNet scorer networks. Two distinct paradigms:
  Era 1 (AV1 + tiny CNN post-filter, 4.06 -> 1.73), Era 2 (neural renderer that
  bypasses the codec, 1.73 -> 1.04 [contest-CUDA A++] after PFP16).
- **Where we are:** contest-CUDA A++ floor `1.043987524793892` (PFP16 A++ on
  Lane G v3 renderer; exact Tesla T4 auth eval). The authoritative artifact is
  `experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/eval/contest_auth_eval.json`;
  stale build-provenance parser fields are quarantined in the final bundle.
- **Live work (Era 3):** Selfcomp paradigm portfolio (8 Modal lanes); not
  reported until [contest-CUDA] verified.
- **Public-frontier forensic note:** PR102 is tracked as a zero-byte
  inference-tuning submission over PR100 archive custody: the PR102 source
  fetches BradyMeighan's `hnerv-lc-v2-archive/archive.zip` by SHA-256, while
  PR102 changes runtime constants only. Treat this as lineage/composition
  evidence, not a local score claim, until exact CUDA replay lands.
- **Why the writeup matters:** the challenge has a separate writeup prize.
  This Ara package is engineered for both human-reader skim and agent-reviewer
  reproduction. It is not an arXiv/preprint commitment.

## How to navigate this artifact

| Layer        | Purpose                                                        | Entry file                          |
|--------------|----------------------------------------------------------------|-------------------------------------|
| `logic/`     | Scientific reasoning, falsifiable claims, dependencies         | `logic/problem.md`                  |
| `src/`       | Index of executable modules and configs                        | `src/index.md`                      |
| `trace/`     | Exploration DAG: decisions, dead-ends, pivots                  | `trace/exploration_tree.yaml`       |
| `evidence/`  | Raw outputs that ground every claim                            | `evidence/index.md`                 |

A claim in `logic/claims.md` carries explicit forensic bindings: code in
`src/index.md`, experiment definitions in `logic/experiments.md`, and
evidence in `evidence/results/...`. Following any claim's binding chain is the
reproducibility contract.

## Provenance and disclosure

- The `disclosure_policy` field above is binding. Public surfaces (the
  Cloudflare site, optional preprint, public PR, or release notes) MUST NOT
  include items under `disclosure_policy.private`.
- This artifact is generated and partially maintained by `tools/ara_compile.py`
  which ingests `.ralph/run_log.md`, `experiments/results/**`, and
  topic-indexed project-memory summaries.

## Relationship to the legacy writeup files

The pre-Ara writeup is preserved as the narrative spine for the human-reader
audience:

- `docs/writeup_draft.md` — long-form prose draft
- `docs/paper_outline.md` — 11-section outline + Era 2/3 addenda
- `reports/graphs/final_writeup_draft.md` — the working Track B writeup
- `reports/writeup_working.md` — daily working notes

The Ara layers re-organize this content into machine-queryable form WITHOUT
overwriting the human-readable drafts. The two coexist; the human reviewer
reads the prose, the agent reviewer queries the layers.

## Compilation date

This manifest was compiled on 2026-04-29 by hand with the intent that
`tools/ara_compile.py` (still in scaffold form) will keep it in sync with the
underlying lab state going forward. See `trace/compilation_log.md`.
