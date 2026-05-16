# L5 / Cathedral Citation And Source-Authority Audit

Date: 2026-05-15

## Rule

Literature and OSS sources route hypotheses. They do not authorize Pact score
claims. Score authority remains byte-closed archive/runtime custody plus exact
evaluation on an explicitly labeled contest axis.

## Current Source Anchors

- Official contest repository:
  <https://github.com/commaai/comma_video_compression_challenge>
  - Score formula: `100 * segnet_distortion + sqrt(10 * posenet_distortion) + 25 * rate`.
  - Submission shape: public PR with archive link plus `inflate.sh`; optional
    compression script and other assets.
  - Evaluation modes: `cpu|cuda|mps`, with official CPU and T4 CUDA runners
    and a 30-minute time limit.
  - Current mirrored leaderboard shows PR101 `hnerv_ft_microcodec` at `0.193`,
    then PR103/PR102/PR100 at `0.195`.

- HNeRV:
  <https://openaccess.thecvf.com/content/CVPR2023/html/Chen_HNeRV_A_Hybrid_Neural_Representation_for_Videos_CVPR_2023_paper.html>
  - Use as a representation-family citation and control-arm context.
  - Pact must still prove export grammar, runtime closure, exact inflate, and
    axis-labeled score.

- NeRV:
  <https://papers.nips.cc/paper/2021/hash/b44182379bf9fae976e6ae5996e13cd8-Abstract.html>
  - Use as the image-wise implicit video representation baseline family.
  - Does not prove contest-specific SegNet/PoseNet behavior.

- HiNeRV:
  <https://papers.nips.cc/paper_files/paper/2023/hash/e5dc475c370ff42f2f96dddf8191a40c-Abstract-Conference.html>
  - Use as hierarchy/encoding evidence for NeRV-family alternatives.
  - Treat reported rate-distortion gains as standard-dataset evidence, not
    Pact score movement.

- Ballé scale hyperprior:
  <https://arxiv.org/abs/1802.01436>
  - Use as the primary hyperprior/side-information reference.
  - Pact requirement: count CDF/table overhead, stream order, runtime decode
    closure, and exact-eval axis separately.

- CompressAI:
  <https://interdigitalinc.github.io/CompressAI/models.html>
  <https://interdigitalinc.github.io/CompressAI/entropy_models.html>
  - Use as API precedent for entropy bottlenecks and Gaussian conditionals.
  - Do not rely on uninitialized or unaccounted entropy-model state.

- Constriction:
  <https://github.com/bamler-lab/constriction>
  - Use as ANS/range-coding OSS reference for low-level PacketIR coders.
  - Pact requirement: include PMF/table/header/runtime costs and decode fuzzing.

- Cool-Chic 5.0:
  <https://arxiv.org/abs/2605.02726>
  - Newly relevant overfitted-codec evidence; reported 2026-05-04.
  - Keep research-only until Pact has a `0.bin` grammar, deterministic decoder,
    timing smoke, and exact-eval archive path.

## Cathedral / L5 Consequence

Cathedral and L5 rankers should carry fields equivalent to:

- `source_supports`
- `paper_claim_scope`
- `pact_must_prove`
- `decode_complexity_evidence`
- axis-specific score fields
- archive SHA and runtime-tree SHA

Predicted rows and literature anchors may influence dispatch priority, but they
must not rank, kill, promote, or submit a lane without the same archive/runtime
custody and contest-axis evidence required by PacketIR exact closure.
