# Contest Submission Packet Pipeline

Use `scripts/build_contest_submission_packet.py` to turn an exact eval artifact
directory into a deterministic metadata packet for submission review. The packet
contains only a JSON manifest and markdown checklist; it does not copy raw
frames, eval work directories, or the archive payload.

Canonical C-059 invocation:

```bash
python3 scripts/build_contest_submission_packet.py \
  --artifact-dir experiments/results/lightning_batch/exact_eval_qzs3_b32_maskfirst_qp1_fix1_t4_20260502T0331Z \
  --expected-archive-sha256 cf44aa7fdb13b9ab6236aefde1f5f58e915d7dfa99128246235443841396c6ab \
  --expected-archive-size-bytes 276347 \
  --expected-samples 600
```

Outputs:

- `submission_packet/submission_packet_manifest.json`
- `submission_packet/submission_packet_checklist.md`

The builder validates:

- `archive.zip` exists and matches the SHA-256 and byte count recorded by
  `contest_auth_eval.json`.
- `contest_auth_eval.json` contains finite score, component, archive byte, and
  sample-count fields, and the contest formula recomputes from components.
- Optional `component_trace.json` agrees with the auth-eval cross-check when
  that cross-check is present.
- Optional `report.txt`, `eval_provenance.json`,
  `contest_auth_eval.adjudicated.json`, and `adjudication_provenance.json` are
  recorded with path, bytes, and SHA-256 when present.
- Optional planner ledgers, visualizations, and next-action tranche documents
  are recorded as non-score supporting artifacts. They are explicitly tagged
  `planning_or_proxy_only`, `visual_audit_only`, or `roadmap_only`, and never
  become packet score authorities.

Evidence grade is field-supported only. The packet records custody fields and
sets `score_claim=false`, `ranking_claim=false`, and `promotion_claim=false`;
use the exact `contest_auth_eval.json` and adjudication process as the score
authority.

Canonical C-067 metadata packet refresh:

```bash
python3 scripts/build_contest_submission_packet.py \
  --artifact-dir experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z \
  --output-dir experiments/results/submission_packet_c067_20260502/automated_packet \
  --score-authority contest_auth_eval.adjudicated.json \
  --expected-archive-sha256 226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a \
  --expected-archive-size-bytes 276214 \
  --expected-samples 600 \
  --planner-ledger experiments/results/c067_yousfi_fridrich_field_equations_20260502/top2_plan_guarded.json \
  --planner-ledger experiments/results/c067_cmg3a_body200_atom_field_20260502/body200_field_plan.json \
  --visualization reports/yousfi_fridrich_observability_20260502/target_gap.svg \
  --visualization reports/yousfi_fridrich_observability_20260502/score_breakdown.svg \
  --next-action-tranche docs/runbooks/contest_faithful_submission_next_tranche_20260502.md
```

The generated packet is a review surface for the current frontier plus
planning/report artifacts. The only score-bearing input in that command is the
adjudicated exact CUDA JSON inside the artifact directory; every other listed
file is metadata-only support.

## Apogee PR Naming And Supplement Conventions

Use `Apogee` as the public submission name and lower-case `apogee` for files,
release manifests, and hosted supplement identifiers.

The upstream PR template fields map to our packet surfaces as follows:

- `submission name`: `Apogee`
- `archive.zip`: `${APOGEE_ARCHIVE_ZIP_URL}` pointing at the exact final
  archive bytes from the A++ artifact directory
- `report.txt`: the final exact T4/equivalent CUDA report text, copied without
  proxy/CPU/MPS substitution
- `GPU required`: `yes`
- `compression script`: include supporting reproduction code when practical,
  but never make it a hidden source of score-affecting payload bytes
- `additional comments`: point to `${LIGHTNING_SUPPLEMENT_URL}`,
  `${CLOUDFLARE_PAGES_URL}`, and `${APOGEE_RELEASE_MANIFEST}` only after those
  URLs have been intentionally sanitized and published

Public support surfaces:

- PR body template: `docs/submission_template.md`
- public supplement runbook:
  `docs/runbooks/apogee_public_supplement_20260502.md`
- Lightning notebook skeleton: `notebooks/apogee_lightning_supplement.ipynb`
- Cloudflare static site bundle: `reports/graphs/public_site/`, generated from
  `reports/graphs/site/` by `reports/graphs/build_public_site_bundle.py`

Before public upload, run the strict public-release hygiene scan over exactly
those surfaces. The scan must pass before any link is copied into the public PR.

## 2026-05-02 Frontier And Guardrail Update

Current frontier wording must point at C-067, not the older C-059 packet:
`experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/contest_auth_eval.adjudicated.json`
is the active A++ score authority for score `0.31561703078448233`, archive
`276214` bytes, SHA-256
`226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`.
The C-059 packet remains lineage custody support and is not a separate score
source.

Submission-pipeline notes must carry these evidence boundaries:

- CMG2 exact T4 results are `A-negative scoped forensic`, not score rows:
  plain 2x2, top512 AMR1 repair, and top256 AMR1 repair all collapsed
  PoseNet/SegNet and have `promotion_eligible=false`.
- The predictive mask-grammar row-span probe is `empirical_byte_probe_only`.
  The `63212` byte row-span payload is a design input for CMG3, not a score or
  rankable archive.
- The next tranche is CMG3 as a closed charged row-span archive: deterministic
  runtime decoder, validator allowlist, archive builder, local smoke, and exact
  CUDA gate before any promotion wording.
- C-067 byte-accounting profiles are observability only:
  `experiments/results/c067_archive_byte_accounting_20260502/archive_byte_accounting.md`
  and
  `experiments/results/c067_self_compression_profile_20260502/archive_byte_accounting.png`
  show a `23454` byte unchanged-distortion gap to sub-`0.300`, stream bytes
  `219472/55965/677`, and `100` bytes ZIP overhead. They explain next-action
  pressure but must stay out of score-bearing rows.
- PMG atomtop4068 is an L40S CUDA `A-negative scoped forensic`, not a packet
  frontier: score `28.41411894150047`, archive `195762` bytes, SHA-256
  `2567dc04185cf20775f1f6c088395aa8df9e4484daa8b25001e940d62a5d6497`,
  PoseNet `62.34251404`, SegNet `0.03315286`, artifact
  `experiments/results/lightning_batch/exact_eval_pmg_hotspot_atomtop4068_l40sdiag_20260502T1445Z/contest_auth_eval.json`.
  Include it only in negative-results/supporting-artifact sections with
  allowed use `mask-grammar redesign input`.
- SJ-KL runtime integration and target-slot closure are production-readiness
  support only. The tensor-prep manifests at
  `experiments/results/sjkl_tensor_prep_c067_smoke_20260502/sjkl_pair_tensor_prep_manifest.json`
  and
  `experiments/results/sjkl_tensor_prep_c067_full_20260502/sjkl_pair_tensor_prep_manifest.json`
  record `target_slot=0` for JointFrameGenerator `fake1`; both are
  `build_tensor_prep_only`, `score_claim=false`, and `promotion_eligible=false`.
- C-067 fixed-slice matrix hardening is planning-only. The refreshed matrix can
  consume recognized `public_pr67_qzs3_qp1_fixed_slices` payloads and records
  slice bytes `219472/55965/677` with `score_claim=false`; it does not change
  the active frontier.
- Every GPU dispatch must have a terminal claim row when it ends:
  `completed_...`, `completed_score=...`, `completed_no_frontier`, or
  `failed_...`. Terminal rows close matching older active rows and prevent
  phantom dispatch conflicts in the shared ledger.
