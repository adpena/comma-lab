# Partner finding validation: PR106 r2 CPU/CUDA and PR101 grammar (2026-05-11)

## Scope

This ledger validates the partner-agent claim:

1. PR106 latent sidecar r2 has a paired CPU result at `0.22809` while CUDA is
   `0.20665`; the pose CPU/CUDA ratio matches r1 closely enough to ratify a
   substrate-class boundary for the latent-sidecar subcluster.
2. PR101 ranked/no-op sidecar grammar saves bytes on PR106 r2 but is
   forensic-only because the current runtime does not consume `format_id=0x02`.
3. PR93 delta-varint pose codec is the top public-PR primitive candidate.

No dispatch was launched. This is evidence review only.

## Validated numeric custody

Artifacts:

- r1 CUDA:
  `experiments/results/modal_auth_eval/pr106_latent_sidecar_20260511T150517Z/contest_auth_eval.adjudicated.json`
- r1 CPU:
  `experiments/results/modal_auth_eval_cpu/pr106_latent_sidecar_20260511T151955Z/contest_auth_eval.json`
- r2 CUDA:
  `experiments/results/modal_auth_eval/pr106_latent_sidecar_r2_20260511T160358Z/contest_auth_eval.json`
- r2 CPU:
  `experiments/results/modal_auth_eval_cpu/pr106_latent_sidecar_r2_20260511T171453Z/contest_auth_eval.adjudicated.json`

Formula recomputation used:

```text
score = 100 * seg_dist + sqrt(10 * pose_dist) + 25 * archive_bytes / 37,545,489
```

Results from the stored JSON fields:

| Anchor | Axis | Score | Recomputed | Seg | Pose | Bytes | Archive SHA-256 |
|---|---|---:|---:|---:|---:|---:|---|
| r1 | CUDA | 0.207394280854 | 0.207394310169 | 0.00064893 | 0.00003281 | 186,808 | `947b85e8a69db295d4dcf80b0b528639c47839f40f289a2c05b70a2064658b48` |
| r1 | CPU | 0.228680284518 | 0.228680313832 | 0.00063766 | 0.00016424 | 186,808 | `947b85e8a69db295d4dcf80b0b528639c47839f40f289a2c05b70a2064658b48` |
| r2 | CUDA | 0.206645885457 | 0.206645986798 | 0.00064260 | 0.00003236 | 186,822 | `7f926bc3e213af1c3ea4be0608c63d041d455eb6b988562b64465e81b25f3a3f` |
| r2 | CPU | 0.228092382711 | 0.228092484052 | 0.00063196 | 0.00016402 | 186,822 | `7f926bc3e213af1c3ea4be0608c63d041d455eb6b988562b64465e81b25f3a3f` |

The recomputation deltas are JSON-rounding scale only (`~3e-8` for r1,
`~1e-7` for r2). CPU/CUDA archive SHA-256s match within each pair, so this is
an apples-to-apples archive/runtime axis comparison.

Derived axis facts:

| Anchor | CPU-CUDA score | CUDA-CPU score | CPU/CUDA pose | CUDA/CPU pose | CPU/CUDA seg |
|---|---:|---:|---:|---:|---:|
| r1 | +0.021286003663 | -0.021286003663 | 5.005791 | 0.199769 | 0.982633 |
| r2 | +0.021446497254 | -0.021446497254 | 5.068603 | 0.197293 | 0.983442 |

Validation verdict:

- **Confirmed:** r2 paired CPU = `0.228092382711`; r2 CUDA = `0.206645885457`.
- **Confirmed:** r2 pose CPU/CUDA ratio = `5.068603`.
- **Adjusted precision:** r2-vs-r1 pose-ratio drift is `1.2548%` from stored
  JSON fields, not exactly `1.2%`. Use "about 1.25%" unless a higher-precision
  artifact supersedes these rounded fields.
- **Confirmed:** latent-sidecar subcluster behavior is opposite of the A1
  HNeRV score-gradient cluster: CUDA wins for r1/r2, while A1 CPU wins. This is
  packet-specific evidence, not a universal CPU/CUDA rule.
- **Evidence grade:** r2 CPU is diagnostic/score-grade axis evidence, not a
  promotion claim. r2 CUDA remains the exact frontier axis for this archive.

## PR101 grammar byte claim

Raw measurement artifact:

`experiments/results/pr106_r2_pr101_pr103_grammar_measurement_20260511T180000Z/`

Structured fields and actual ZIP bytes validate:

- baseline PR106 r2 archive: `186,822` bytes
- optimized PR101-grammar archive:
  `experiments/results/pr106_r2_pr101_pr103_grammar_measurement_20260511T180000Z/sidecar_archive_pr101_grammar.zip`
- optimized archive SHA-256:
  `c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383`
- optimized archive bytes: `186,780`
- net archive delta: `-42` bytes
- payload-only sidecar delta: `-48` bytes
- rate-only score delta:
  `(186780 - 186822) * 25 / 37,545,489 = -0.0000279661`

The ignored raw artifact contains one stale explanatory string that says
`186822 - 186786` and `-2.4e-5`; the structured fields and actual ZIP size
prove the correct net delta is `42` bytes and `-2.8e-5`. Do not edit the ignored
raw artifact; this ledger supersedes that prose.

Current-runtime parser validation:

```text
first_bytes fe0213d70200ff26
ValueError sidecar format_id mismatch: got 0x02, expected 0x01
```

Validation verdict:

- **Confirmed:** PR101 ranked/no-op sidecar grammar is a real byte saving on
  r2.
- **Confirmed:** it is not a score-lowering candidate yet because
  `submissions/pr106_latent_sidecar_r2/inflate.py` accepts only `format_id=0x01`.
- **Classification:** forensic measured primitive; `score_claim=false`,
  `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false` until a
  paired runtime decoder consumes `format_id=0x02` and a no-op detector proves
  closure.

## Public PR primitive count claim

Evidence checked:

- tracked review:
  `.omx/research/nonhnerv_residual_basis_and_pr_backlog_clean_pass_review_20260511.md`
- ignored backlog:
  `experiments/results/public_pr_nonhnerv_mechanism_backlog_20260511T171636Z/backlog.jsonl`
- ignored synthesis:
  `experiments/results/public_pr_nonhnerv_mechanism_backlog_20260511T171636Z/synthesis.md`

Findings:

- `backlog.jsonl` currently has **7 JSONL rows**, not 8.
- `synthesis.md` says "8 typed rows" and "30+ reusable primitives".
- PR93 delta-varint pose codec is documented as rank 1 by EV/byte at the PR106
  r2 operating point and is plausibly a small port target
  (`tac.packet_compiler.pr93_delta_varint_pose`).

Validation verdict:

- **Confirmed:** PR93 delta-varint pose codec is the current top-ranked public
  PR primitive candidate in the synthesis.
- **Not confirmed as written:** "8 new primitives identified" is imprecise.
  Current evidence is a 7-row external-mechanism backlog plus a 30+ primitive
  candidate inventory. Treat it as planning metadata, not a measured primitive
  landing.
- **Dispatch status:** no primitive from that backlog is exact-eval ready until
  it has a canonical `tac.packet_compiler` port, golden vectors, runtime
  consumer, byte/semantic no-op proof, lane claim, and paired exact eval.

## Deterministic reproducibility gate

Future updates to this cluster should preserve:

- exact artifact path, byte count, SHA-256, device axis, sample count, and
  score components;
- formula recomputation from components rather than display-rounded scores;
- CPU/CUDA pair comparisons only when archive SHA-256s match;
- runtime parser proof before any byte-changed archive is allowed to move from
  forensic to candidate;
- immutable raw artifacts with tracked supersession ledgers for discovered
  prose mistakes.

