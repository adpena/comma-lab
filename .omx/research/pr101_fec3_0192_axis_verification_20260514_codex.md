# PR101 FEC3 0.192097 axis verification - 2026-05-14

## Classification

The `0.19209788683213053` result is a legitimate recovered Linux
`[contest-CPU]` reproduction artifact for the PR101/FEC3 compact selector
packet. It is not a `[contest-CUDA]` result, not promotion-ready, and not a
public leaderboard score claim.

## Custody

- eval JSON:
  `experiments/results/modal_auth_eval_cpu/archive_8866ebb655e9/contest_auth_eval.json`
- eval JSON sha256:
  `ec2a2ad7a053d7d43a7209e8a583d0181fdee787c5fb9169002d04871091d586`
- packet archive:
  `experiments/results/pr101_frame_exploit_selector_fec3_compact_exact_k8_cpu_overlay_20260514_codex/archive.zip`
- archive bytes: `178517`
- archive sha256:
  `8866ebb655e96ccf0ffcd84feae08c131734cba8c402bfb8c661a29f289ce409`
- runtime tree sha256:
  `7e28ecabc5cc6c21494e3cd0178b112244a43bc72cf116c0bfdd950492f4ebb4`
- runtime content tree sha256:
  `60ccb8066f6c6ddf5983fed6fbdbc8bb679ab47d1af6387ae22fc8874b1bd6b5`
- samples: `600`
- platform: Linux x86_64 CPU via Modal recovery

## Components

- `avg_segnet_dist`: `0.00056029`
- `avg_posenet_dist`: `0.00002959`
- seg contribution: `0.056028999999999995`
- pose contribution: `0.01720174409761987`
- rate contribution: `0.11886714273451067`
- canonical formula score:
  `0.19209788683213053`
- display-rounded score field: `0.19`

The formula check is:

```text
100 * 0.00056029
+ sqrt(10 * 0.00002959)
+ 25 * 178517 / 37545489
= 0.19209788683213053
```

## Axis guard

The JSON correctly marks:

- `score_axis`: `contest_cpu`
- `evidence_grade`: `contest-CPU`
- `evidence_semantics`: `public_leaderboard_cpu_reproduction`
- `cpu_leaderboard_reproduction_eligible`: `true`
- `exact_cuda_eval_complete`: `false`
- `score_claim`: `false`
- `score_claim_valid`: `false`
- `promotion_eligible`: `false`
- `rank_or_kill_eligible`: `false`

This means the artifact is useful as CPU-axis signal and CPU/CUDA drift
diagnosis only. It must not be promoted as a CUDA frontier result.

## Selector / grain / water-fill exhaustion answer

The answer is no: this space is not exhausted.

What is correctly engineered:

- FEC3 charges the compact selector palette in archive bytes.
- The packet is byte-closed and replayed on Linux CPU.
- The packet keeps score-claim and promotion flags false until CUDA evidence.

What remains open:

- It misses the operator `<0.192` CPU threshold by about `9.8e-5`.
- CUDA/T4 is unmeasured for this packet.
- `experiments/results/frame_exploit_cuda_transfer_audit_20260514_agent_codex/audit.json`
  records CPU/MPS-to-CUDA transfer failure for earlier frame-exploit selector
  paths, so CPU/MPS positives cannot be inferred to CUDA.
- HDM8 film-grain/water-fill artifacts remain proxy/planning rows until a
  byte-closed selector archive and exact CUDA replay exist.

Next score-lowering work should be CUDA-safe, byte-closed selector/grain
engineering, not promotion of the CPU near-miss.

## 2026-05-15 CUDA Axis Closure

The matching detached Modal T4 CUDA eval completed after the CPU-axis
verification.

Artifacts:

- output dir:
  `experiments/results/modal_auth_eval/pr101_fec3_compact_exact_k8_cuda_20260515T000000Z`
- Modal call id: `fc-01KRMFR42NV1XJAAGSA7YHDT59`
- eval JSON:
  `experiments/results/modal_auth_eval/pr101_fec3_compact_exact_k8_cuda_20260515T000000Z/contest_auth_eval.json`
- eval JSON sha256:
  `31c47970ce708de7d13268e83b940f6111459119dff0f5548beac99b41ec51de`
- result JSON sha256:
  `ee37d60672eea9f1789c26a5863c017ed751337f1172b7fee697a98f7883bd9b`
- packet archive:
  `experiments/results/pr101_frame_exploit_selector_fec3_compact_exact_k8_cpu_overlay_20260514_codex/archive.zip`
- archive bytes: `178517`
- archive sha256:
  `8866ebb655e96ccf0ffcd84feae08c131734cba8c402bfb8c661a29f289ce409`
- uploaded runtime tree sha256:
  `40bfaea67aa506e27b763452c2871dd43fba174e4d6680e57a91801f1f5fa923`
- samples: `600`
- platform: Linux x86_64 T4 via Modal recovery

CUDA components:

- `avg_segnet_dist`: `0.00066299`
- `avg_posenet_dist`: `0.00016893`
- canonical formula score:
  `0.22626723761043824`
- evidence grade: `[contest-CUDA]`
- exact CUDA complete: `true`
- score claim: `true`
- promotion eligible: `false`

Classification:

- The `0.19209788683213053` value is confirmed legitimate only on the
  `[contest-CPU]` axis.
- The same byte-closed archive on `[contest-CUDA]` scores
  `0.22626723761043824`, so it is not a public leaderboard breakthrough and
  not a sub-0.192 CUDA candidate.
- The CPU/CUDA gap for this archive is `0.034169350778308`.
- Relative to the prior PR101 CUDA anchor around `0.22650343150032118`, this
  selector is still a small CUDA improvement (`-0.000236193889883`), so the
  frame-exploit/selector family is not a no-op. The failure is magnitude and
  CUDA transferability, not archive engineering.

Updated answer to the operator question:

- The `0.192` was a real CPU-axis result, not a hallucinated proxy.
- It is not leaderboard-legit for the CUDA contest axis.
- The FEC3 packet engineering is correct enough to charge the selector bytes,
  preserve runtime custody, run exact CPU, and run exact CUDA.
- The current K8 selector/grain/water-fill basin is not exhausted: it improved
  PR101 CUDA slightly, but the CPU-winning modes do not transfer enough through
  CUDA PoseNet/SegNet. Future work must be CUDA-in-loop, vectorized/sharded,
  and byte-closed from the start.
