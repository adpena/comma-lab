# DAG FEED — DDM CC3 mixed-coder receiver integration

Date: 2026-07-25
Lane: `lane_ddm_cc3_mixed_coder_receiver_integration_20260725`
Status: `MEASURED_ADVISORY_MAIN_REVIEW_REQUIRED`

## Inputs

- CC2 SSD receipt SHA-256
  `f9432959d9c8711276379ef681f5b6985157f49bdd4b2f4a401bfb35ce737ec1`
- source composition
  `139538 B / 08d03f75f818b22c9a9a6aad33c6f879001743f80b66d71fe1fc9b3a094567a2`
- LP1 receipt SHA-256
  `6bd6a5baaa8f5995e93ef594e880beac77e9aa2b2083e661598c84feaba13fd5`

## Transform

`ddm_cc3_mixed_coder_receiver.build_mixed_archive` recursively replaces
only the eight negative CC2 leaves, preserving raw bytes for the other 19
leaves and preserving nested member identity, order, metadata, and the
W_joint trailing receiver suffix. The free runtime restores all frames
before entering the existing E3/E4/E5 WS1 plus PC1 receiver.

## Measured outputs

- counted archive:
  `136116 B / ba18024211ff2e1de189d1a094b157c63aac86b21dca2dbce331e4385e49aebe`
- delta: `-3422 B`; integration overhead `0 B`; falsifier `PASS_PAYS`
- codecs: `1 G4`, `7 Bellard KT-mixing`
- exact restoration: `27/27` physical leaves; selected `8/8` frames
- fresh CC3 replay: `135/135` canonical CC2 frames parse back;
  replay core SHA-256
  `b392dab369ec2a257a29a1ed9beeb59c4773e35df2af7a76eeb6a4e38538a107`
- locked inflate: `489.727193 s`, under 1800 seconds
- candidate and raw-source control:
  `3662409600 B / 5094e277dc4c736ad1ab50aead9f49630319bb6e3d42c48e9777fbdd09c215f3`
- n600 batch32 endpoint:
  `d_seg=0.024731920030381944`,
  `d_pose=163.0492342914382`
- LP1 coordinated total: `130789 B` (`DERIVED_COORDINATED_BUDGET`)

## Feed edges

1. `CC2 price table -> CC3 exact recursive archive builder`
2. `CC3 counted members -> canonical E3/E4/E5 inflate launcher`
3. `CC3 restoration -> W_joint receiver -> PC1 receiver -> uint8 raw`
4. `raw identity -> zero distortion costates`
5. `measured -3422 B -> LP1/C1 coordinated accounting`

The 135-frame replay receipt is
`.omx/research/ddm_cc3_135_frame_replay_receipt_20260725.json`.

The fresh full-receiver endpoint falsifies CC2's instance-specific
`IDENTITY_ACTIVE_ALL_QUANTIZED_COORDINATES_ZERO` scorer-reuse premise.
Consumers must use the fresh endpoint for this exact PC1 composition while
retaining the measured zero-distortion mixed-vs-raw rate delta.

No dispatch, promotion, contest score, live campaign mutation, or pointer
movement is authorized. MAIN review is required before landing.
