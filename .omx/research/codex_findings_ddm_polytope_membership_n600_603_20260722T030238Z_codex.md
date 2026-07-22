---
title: Codex adversarial review of DDM polytope membership and n600 closure
utc: 2026-07-22T03:02:38Z
task: 603
lane_id: lane_ddm_polytope_membership_n600_603_20260722
review_round: 1
main_landing_review_required: true
---

# Review disposition

`PASS_WITH_SCOPED_NEGATIVE`. The implementation and receipt support a local frozen-SegNet
membership curve and scorer-free n600 archive-closure claim. They do not support `d_seg`, `d_pose`,
contest score, candidate status, promotion, or current-grammar all-class efficacy.

# Re-derived checks

- Authority SHA-256 and byte count matched before work; `execution_allowed=false` is compiled and
  no dispatch/provider/GPU surface is reachable.
- Target and described logits are evaluated separately with the frozen upstream SegNet, canonical
  batch size 16, deterministic CPU-Torch algorithms, four threads, and SHA-bound source/weights.
- The 38,502,892-byte weight file is local custody only and absent from the 2,565,528-byte archive.
- n64 and n256 target/description batches retain at most one source chunk and one scorer batch.
- n600 describes and decodes 600 pairs from the same archive with at most two combined chunks,
  deterministic compile/decode x2, parse/re-encode identity, exact pair coverage, six consumed
  streams, unique-home coverage equal to archive bytes, and resume-identical terminal state.
- Primary checkpoints for all three stages and the resumed terminal checkpoint are preserved.
- Focused and predecessor verification is 45 passed; Ruff, Python compilation, and diff checks are
  clean. Source/tool review entities received clean review passes before serializer commit.

# Adversarial findings

1. **CONFIRMED proxy failure:** n256 RGB-channel disagreement is 0.229040582975 but real cell escape
   is 0.505538980166. The old diagnostic under-reports by 0.276498397191 and must not carry SegNet
   authority.
2. **CONFIRMED class collapse:** n256 membership 0.494461019834 is dominated by Undrivable
   (0.999996584559). Road and Lane are zero, and MyCar is 0.000011158723. The measurement rung is
   valid; the current grammar is not an adequate all-class member solver.
3. **CONFIRMED exactness is the wrong scalar objective:** n256 RGB-pixel exactness is only
   0.001795709133, yet membership is 0.494461019834. The relationship is stratum-dependent, with
   only 0.129902776900 membership on boundary sites versus 0.502304952747 in cell interiors.
4. **SCOPED custody gap retained:** fresh target cells match cached `lstars` at 0.999844054381, not
   exactly 1. The receipt uses same-run target cells for its numerator and records the cache gap;
   it does not falsify the retained target or silently claim equivalence.
5. **NO score implication:** n600 closure deliberately omits the scorer. Pose-code completeness is
   storage completeness only. The archive remains `not_a_candidate` and the frontier pointer is
   unchanged.

# Required MAIN landing review

MAIN should verify the cache-crosscheck interpretation, review the large machine-readable stage
receipts, and confirm the supplemental membership row remains outside the inherited 19-row PRIMARY
count. Any future optimizer should target per-class/per-margin membership debt rather than the
headline aggregate, which is confounded by class-basin collapse.
