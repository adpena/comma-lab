# Paper Outline

**Working title:** Evidence-Gated Neural Archive Compression for a Two-Scorer Video Challenge

**Target surfaces:** contest writeup, reproducibility report, arXiv-style systems note, submission packet.

**Current strict frontier for latest packets:** PR100 HNeRV-LC-v2 Apogee
follow-up adapter replay, exact Tesla T4 A++ score `0.22826947142244708`,
archive `178981` bytes, SHA
`afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641`,
runtime tree SHA
`ef6323533666c9cac1c204a9d3f7054157d44a185b16fc859fb3f0438ccd1832`.

**PR100 adapter custody packet:** `experiments/results/submission_packet_pr100_adapter_20260504/submission_packet_manifest.json` and `experiments/results/submission_packet_pr100_adapter_20260504/submission_packet_checklist.md`. The upload directory is `experiments/results/submission_packet_pr100_adapter_20260504/apogee_pr100_hnerv_lc_v2_adapter`.

**Requested narrative comparison anchor:** PR #100 is the public-source
attribution anchor for the current follow-up packet. PR98/PR107, PR99, PR95
stem-permutation, exact public PR95 replay, and conservative PR95 repack are
retained as superseded exact predecessors; PR101-PR106, PR96, and PR91/HPM1
remain external until exact CUDA replay lands.

## 1. Abstract

- Define the contest objective: minimize SegNet distortion, PoseNet distortion, and archive bytes under `archive.zip -> inflate.sh -> upstream/evaluate.py`.
- State the verified result from exact evidence: PR100 adapter replay for
  strict latest packets, with PR98/PR107, PR99, PR95 stem-permutation,
  PR85-family rows, and C-067 as predecessor/context rows.
- State the method contribution: public semantic-bundle intake, deterministic archive adapters, typed payload custody, exact runtime-tree hashing, dispatch-claim discipline, hygiene scanning, and evidence-gated promotion.
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
4. **Semantic-bundle era:** PR85-family exact replays and recodes moved the frontier below `0.26`.
5. **HNeRV deadline era:** public PR95 exact replay reached `0.23098329465634826`; PR95 stem-permutation reached `0.23089404465634825`; PR100 adapter replay reached `0.22826947142244708`.
6. **Meta-game era:** late public PRs changed the active representation family while exact replay and adapter hardening decided which public claims were evidence.
7. **Unlimited-compute probe era:** inflate-time TTO, scorer-gradient probes,
   and long compression runs informed the atom ledger and production story, but
   do not rank unless all payloads are charged and exact CUDA replay validates
   the final archive.

## 4. Methods

- Archive contract: every score-affecting byte lives in `archive.zip`; no host-local files, network fetches, scorer patches, hidden sidecars, malformed ZIP dependence, or script-embedded payload transfers.
- Representation contract: public-floor mask/renderer/pose geometry becomes typed payload atoms.
- Packing contract: QZS3/QP1/PR64-like containers are deterministic, parsed by structure, and exact-evaled as complete archives.
- Search contract: pose edits are proposal-time optimization only until accepted into a charged archive and exact CUDA eval passes.
- Meta-Lagrangian contract: every atom is priced by bytes, SegNet, PoseNet, runtime, custody risk, and compliance risk. Water-fill, hard-pair, active-subspace, and public-anatomy policies guide proposals; exact CUDA archive evidence is the only promotion signal.
- Hardening contract: deterministic ZIPs, payload closure, scorer-load guards, zip-slip checks, hidden-file/resource-fork exclusion, component gates, dispatch claims, runtime tree hashes, public-release hygiene scans, and JSON adjudication are part of the reproducible system.
- AI-assisted research contract: Grand Council and Skunkworks Council sessions
  are proposal/review generators with named expert roles, not evidence. Their
  outputs enter the ledger as hypotheses, objections, and action plans; exact
  CUDA artifacts remain the promotion gate.
- Report contract: all writeup tables are generated from evidence-tagged claim rows.

## 5. Results

Primary table: PR100 adapter replay, PR99 adapter replay, PR95 stem-permutation repack, PR95 conservative repack, PR95 public exact replay, PR85+STBM/RMB1, PR85, PR84, PR81, C-067, and historical PFP16. Only exact rows with `A++`/`A` can appear in the ranked table.

External context table:

- PR #95: public source attribution anchor; exact local replay is score evidence only for the replayed archive bytes and runtime tree.
- PR #100: public body score is external; local exact T4 replay is the ranked
  score-bearing row.
- PR #101-#106: later public/title claims are urgent deconstruction targets,
  but external until exact CUDA replay lands.
- PR #96: public score is external until exact CUDA replay lands.
- PR #91: public score is external; local replay failed before score in HPM1 entropy decode.
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
| `public_external_context` | PR96/PR95 body score, PR91, PR67/PR65/PR63/PR64 comparisons | `external` |
| `quarantined_exploit_context` | PR68/PR69/PR70 and rule-boundary cases | `invalid`, `external_quarantine` |
| `component_ablation` | Byte/component deltas tied to exact rows | `A++`, `A`, `A-negative`, `empirical` with caveat |
| `negative_results` | Scoped failures and revival conditions | `A-negative`, `B`, `invalid`, `empirical` |
| `submission_checklist` | Payload closure, deterministic ZIP, eval path, CUDA/T4, review | `engineering_policy` |
| `next_wave_roadmap` | Charged pose-basis, hard-pair windows, residual, and packer/layout atoms blocked on exact archives | `prediction`, `empirical` until exact CUDA eval |

Negative rows should distinguish exact measured regressions, invalid compliance lessons, empirical/proxy blockers, and no-op packer bugs. Each row needs an artifact path, failure class, allowed use, and reactivation criterion.

## 7. Figures And Tables

- Score trajectory with evidence grade labels, not ungraded proxy scores, and a late-meta annotation for Quantizr PR #53/#55.
- Rate/component decomposition for PR100 versus PR98, PR99, PR95 stemperm,
  PR95 conservative repack, and public replay.
- Archive payload diagram for charged PR100 HNeRV-LC-v2 packet.
- Meta-Lagrangian atom-waterfill diagram: archive atoms, marginal rate/distortion, runtime/custody constraints, and exact-eval promotion.
- Compliance quarantine diagram showing why script-side payloads are invalid.
- Negative-result table with scope of retirement and revival condition.
- Atom compiler/water-fill diagram showing proposal feedback, charged-byte accounting, and exact-eval promotion gates.

## 8. Limitations

- PR100 adapter replay is below PR95 stemperm under exact T4 custody; public body/CPU scores remain non-ranking external context.
- Public PR text is not local proof.
- Exploit submissions show evaluator loopholes, not contest-faithful compression results.
- Further sub-`0.22826947142244708` claims require exact charged archive evidence.
- OSS and production-readiness claims are limited to committed interfaces, deterministic archive contracts, and hardening checks already present in the measured pipeline.
- Hidden-gem lanes remain open, not exhausted: GP/RAFT pose, hyperbolic foveation, Cool-Chic/C3, SIREN/NeRV, hardware FP8, SJ-KL, wavelets, and learned atom allocation all require exact archive integration before ranking.

## 9. Reproducibility Appendix

Include for every score-bearing row:

- archive path, bytes, SHA-256
- eval JSON path and command
- device and full sample count
- SegNet, PoseNet, rate, reported score, recomputed score
- manifest and payload closure status
- source ledger and review status
