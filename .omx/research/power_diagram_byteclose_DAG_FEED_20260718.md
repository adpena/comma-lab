# FEED-543 power-diagram byte-close (2026-07-18)

`research_only=true` · factor 6 · Task #543 · pointer UNMOVED · MAIN review required

## Executed path

```text
pinned gt_n600.gt_f1 + cached L* + frozen CPU-Torch SegNet
  -> canonical batch-1 frames, ascending order
  -> final-head pre-hook
  -> rank-4 quotient convolution
  -> positive-control fork
       -> frozen CPU-Torch argmax versus cached L*: exact through blocker pixel
       -> serialized-f32 PDW1 target via generic-f64 power_scores:
            BLOCKED at frame 195, pixel (214,112), one mismatch
  -> preserve SSD cache + atomic blocked checkpoint
  -> PASS-1 containment
       -> preserve exact historical source in deterministic non-source gzip
       -> manifest binds container bytes and in-memory original bytes/hash
       -> direct Python invocation fails without mutation
       -> live measurement path becomes fail-closed tombstone
       -> read-only evidence helper has no resume/certify/cleanup surface
       -> receipt outputs confined to existing worktree .omx/research/*.json
  -> post-hoc committed prefix 0..194 only
       -> float64 ridge sufficient statistics
       -> fitted global power target
       -> strict PDW1 parse-back identity
       -> 899,388 / 38,338,560 feature-pullback mismatches
       -> 314 B raw / 306 B Brotli-Q11 target payload
       -> 257 B optimistic rounded-up ideal order-0 entropy estimate
            (free PMF; zero model/header/termination overhead)
  -> governed one-frame reproduction at (195,214,112)
       -> CPU-Torch class 0; cached L* class 0
       -> generic-f64 power class 1
       -> native-f32 power exact tie; first-max class 0
       -> every verified input re-hashed unchanged after inference
  -> NON-EQUIVALENT rate comparison only
  -> BLOCKED: receiver arithmetic unspecified
  -> BLOCKED: spatial quotient field absent
  -> BLOCKED: RGB inverse / uint8 / shared resize / Pose interaction absent
  -> factor 6 remains HAVE/PARTIAL
  -> equation candidate HELD_NOT_REGISTERED
  -> score pointer UNMOVED
```

## Verdict scope

The negative token is
`FROZEN_HEAD_FLOAT32_POWER_TARGET_POSITIVE_CONTROL_BLOCKED_AT_FRAME_195`.
It applies to this serialized-float32 target under the current generic-float64
assignment formulation. It does not kill the real-arithmetic affine/power law,
the power-diagram family, or the witness paradigm.

## Triality disposition

- DAG leg: this FEED, backed by the blocked checkpoint, current post-hoc
  receipt, governed frame-195 diagnostic receipt, original-launch custody, and
  deterministic historical-source container/manifest.
- Equation leg: `affine_head_power_diagram_generator_duality_v1` remains a
  temporary candidate. Real-arithmetic structure is derived; receiver-level
  byte-close parity is not confirmed.
- DSL leg: N/A for the advisory target/diagnostic. A future spatial/RGB
  receiver must be typed, resume-registered, and parser-closed before launch.

## Consumer debt

The next legitimate consumer is not a rate allocator. It is a receiver-contract
probe with two explicit modes—native float32 first-max and current generic
float64 assignment—tested on the frame-195 pixel and then arbitrated by the
actual legal decoder arithmetic. Only after that contract is fixed may a new
governed n600 extraction use fresh scratch. The preserved blocked artifacts
must remain immutable evidence.
