# Paper Outline

**Working title:** Evidence-Gated Neural Archive Compression for a Two-Scorer Video Challenge

**Target surfaces:** contest writeup, reproducibility report, arXiv-style systems note, submission packet.

**Current strict frontier for latest packets:** C-059 QZS3 B32 mask-first QP1 fix1 byte frontier, exact Tesla T4 A++ score `0.3157055307844823`, archive `276347` bytes, SHA `cf44aa7fdb13b9ab6236aefde1f5f58e915d7dfa99128246235443841396c6ab`.

**C-059 custody packet:** `experiments/results/submission_packet_c059_20260502/submission_packet_manifest.json` and `experiments/results/submission_packet_c059_20260502/submission_packet_checklist.md`. The packet is metadata-only and supports custody; it is not a separate score source and records `score_claim=false`, `ranking_claim=false`, and `promotion_claim=false`.

**Requested narrative comparison anchor:** PR #67 remains an external contest-faithful target reporting rounded `0.31`; C-057 and C-058 are retained as immediate internal predecessor rows for the PR67 comparison chain.

## 1. Abstract

- Define the contest objective: minimize SegNet distortion, PoseNet distortion, and archive bytes under `archive.zip -> inflate.sh -> upstream/evaluate.py`.
- State the verified result from exact evidence: C-059 for strict latest packets, with C-057/C-058 as predecessor/context rows.
- State the method contribution: public-floor-basin reconstruction, charged QZS3/QP1 packing, atom compiler/water-fill selection, pose-manifold search, deterministic archive custody, and evidence-gated promotion.
- State the boundary: public PR claims and exploit submissions are external context, not our score evidence.

## 2. Problem And Evidence Standard

- Score formula:

```text
score = 100 * seg_dist
      + sqrt(10 * pose_dist)
      + 25 * archive_bytes / 37,545,489
```

- Evidence grades: `A++`, `A`, `A-negative`, `B`, `empirical`, `derivation`, `prediction`, `external`, `invalid`.
- A++ requires exact archive, SHA/bytes, payload closure, CUDA/T4/equivalent proof, full `600` samples, component recomputation, inflate-budget evidence, and review status.
- CPU/MPS/proxy, byte-only, public PR text, H100-only diagnostics, and exploit/sidecar submissions cannot rank or promote. <!-- MPS-DECISION-WAIVED: this is the rule itself, restated; not an MPS-derived decision -->

## 3. System Arc

1. **Post-filter era:** learned task-aware filtering established scorer-aware optimization but is no longer frontier evidence.
2. **Renderer/PFP16 era:** neural renderer and deterministic pose payloads established exact archive custody and exact CUDA discipline; PFP16 is now historical A++.
3. **Public-floor-basin era:** PR63/PR64/PR67 anatomy motivated repo-built QZS3/QP1 and PR64-style packing in charged archives.
4. **C-057/C-058/C-059 frontier:** anisotropic pose-basis continuation moved the exact internal frontier to `0.3157562807844823`; successive byte-frontier updates moved the strict packet anchor to C-059 at `0.3157055307844823`.

## 4. Methods

- Archive contract: every score-affecting byte lives in `archive.zip`; no host-local files, network fetches, scorer patches, hidden sidecars, malformed ZIP dependence, or script-embedded payload transfers.
- Representation contract: public-floor mask/renderer/pose geometry becomes typed payload atoms.
- Packing contract: QZS3/QP1/PR64-like containers are deterministic, parsed by structure, and exact-evaled as complete archives.
- Search contract: pose edits are proposal-time optimization only until accepted into a charged archive and exact CUDA eval passes.
- Atom compiler contract: candidate mask, renderer, pose, residual, and packer edits carry byte cost, predicted component effect, rejection reason, and archive identity. Water-fill and hard-pair policies guide proposals; exact CUDA archive evidence is the only promotion signal.
- Hardening contract: deterministic ZIPs, payload closure, scorer-load guards, zip-slip checks, hidden-file/resource-fork exclusion, component gates, dispatch claims, and JSON adjudication are part of the reproducible system.
- Report contract: all writeup tables are generated from evidence-tagged claim rows.

## 5. Results

Primary table: C-059, C-058, C-057, C-056, C-053/C-054, C-052, C-051, historical PFP16. Only exact rows with `A++`/`A` can appear in the ranked table.

External context table:

- PR #67: open PR, reports rounded `0.31`, `276564` bytes, PoseNet `0.00048597`, SegNet `0.00061000`; evidence tag `external`.
- PR #65: external postprocess/side-channel signal; evidence tag `external`.
- PR #63/#64: public-floor lineage; local reruns may be diagnostic, but public claims remain `external`.

Exploit quarantine table:

- PR #68: closed proof-of-concept moving payload bytes into script-side data; tag `invalid`.
- PR #69: open boundary refactor with no filled score at inspection time; tag `external_quarantine`.
- PR #70: reports rounded `0.19` but states bytes were moved from `archive.zip` into `inflate.py`; tag `invalid`.

## 6. Automated Submission And Report Sections

The report generator should emit these sections:

| Section | Purpose | Allowed evidence |
|---|---|---|
| `frontier_summary` | Current exact frontier and supersession chain | `A++`, `A` |
| `exact_artifact_table` | Archive/eval rows with SHA, bytes, components, recomputed score | `A++`, `A`, `A-negative` |
| `public_external_context` | PR67/PR65/PR63/PR64 comparisons | `external` |
| `quarantined_exploit_context` | PR68/PR69/PR70 and rule-boundary cases | `invalid`, `external_quarantine` |
| `component_ablation` | Byte/component deltas tied to exact rows | `A++`, `A`, `A-negative`, `empirical` with caveat |
| `negative_results` | Scoped failures and revival conditions | `A-negative`, `B`, `invalid`, `empirical` |
| `submission_checklist` | Payload closure, deterministic ZIP, eval path, CUDA/T4, review | `engineering_policy` |
| `next_wave_roadmap` | Charged pose-basis, hard-pair windows, residual, and packer/layout atoms blocked on exact archives | `prediction`, `empirical` until exact CUDA eval |

Negative rows should distinguish exact measured regressions, invalid compliance lessons, empirical/proxy blockers, and no-op packer bugs. Each row needs an artifact path, failure class, allowed use, and reactivation criterion.

## 7. Figures And Tables

- Score trajectory with evidence grade labels, not ungraded proxy scores.
- Rate/component decomposition for C-059 versus PR67 public fields.
- Archive payload diagram for charged QZS3/QP1.
- Compliance quarantine diagram showing why script-side payloads are invalid.
- Negative-result table with scope of retirement and revival condition.
- Atom compiler/water-fill diagram showing proposal feedback, charged-byte accounting, and exact-eval promotion gates.

## 8. Limitations

- C-059 is below the merged public `0.32` rounded band but remains above PR67's reported external `0.31` claim.
- Public PR text is not local proof.
- Exploit submissions show evaluator loopholes, not contest-faithful compression results.
- Sub-`0.30` remains a target requiring exact charged archive evidence.
- OSS and production-readiness claims are limited to committed interfaces, deterministic archive contracts, and hardening checks already present in the measured pipeline.

## 9. Reproducibility Appendix

Include for every score-bearing row:

- archive path, bytes, SHA-256
- eval JSON path and command
- device and full sample count
- SegNet, PoseNet, rate, reported score, recomputed score
- manifest and payload closure status
- source ledger and review status
