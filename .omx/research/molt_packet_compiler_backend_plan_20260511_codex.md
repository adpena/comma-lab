# Molt as PacketIR/native-byte-transducer backend (2026-05-11)

## Decision

`adpena/molt` is an appropriate backend candidate for the deterministic packet
compiler, not a replacement for the Python reference oracle or contest auth-eval
harness.

The useful target is a small native lowering path for fixed byte transducers:

- PR101 ranked sidecar / centered-delta / split-Brotli wrappers;
- PR103 merged range streams and latent-hi arithmetic wrappers;
- PacketIR identity / canonicalize / optimize passes once the grammar is
  explicit;
- fast preflight scanners that operate on one indexed file pass;
- optional tiny inflate helpers only after byte-for-byte conformance vectors
  pass on Python and native implementations.

## Guardrails

- Python remains the reference implementation for new math and exact-eval
  custody.
- Molt output is promotable only after golden-vector parity on canonical bytes,
  malformed-negative vectors, deterministic binary provenance, and strict
  packet closure.
- No scorer loads, provider assumptions, hidden sidecars, or local paths in a
  native helper.
- Small binary size is a real advantage only when it replaces runtime overhead
  or produces charged-byte savings. Native speed alone is not a score claim.
- Contest targeting must declare `target_mode`: `contest_one_video_replay`,
  `contest_generalized`, or production-only.

## First implementation tranche

1. Export the current Python PR101/PR103 packet-compiler golden vectors as the
   cross-language contract.
2. Build a Molt proof binary that decodes/encodes one PR101 centered-delta
   vector and one PR103 latent-hi vector byte-for-byte.
3. Add a native-backend manifest with toolchain hash, source hash, binary hash,
   byte count, and command.
4. Compare binary size and startup time against Python helper invocation.
5. Only then consider embedding the native helper in a contest runtime packet.

## Score-lowering relevance

The near-term score path is still PR106 R2 exact paired CPU/CUDA plus yshift /
residual sidecar search. Molt becomes score-relevant when PacketIR can express a
measured-positive transform and the native helper either reduces charged runtime
bytes or makes exhaustive transform search cheap enough to find a better packet.

Until then, Molt is a dev-velocity and compiler-hardening axis.
