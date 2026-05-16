# TT5L AC-state entropy-claim demotion - 2026-05-16

Scope: Time-Traveler L5 / L5-v2 paper-fidelity and section-role custody.

Finding: TT5L v1 consumes `ac_state_blob`, but it consumes it as deterministic
residual-calibration state. It is not yet a range, arithmetic, or ANS entropy
decoder. Prior parser-role labels exposed the section as
`entropy_model_or_range_stream`, which was stronger than the implementation and
could give the L5-v2 staircase false paper or MDL authority.

Change:

- `TT5L_SECTION_ROLES["ac_state_blob"]` is now `sidecar_or_correction_stream`.
- TT5L archive docs now state that v1 AC-state is residual calibration, not
  range/ANS entropy decode.
- Packet-section and parser tests assert the demoted role.
- The substrate composition matrix records this limitation in
  `decode_complexity_evidence`.

Status: no score claim, no promotion claim. Re-enable an entropy-model role only
after a real range/ANS decode path lands with byte-mutation consumption proof,
full-frame inflate parity, T4 timing, and paired contest CPU/CUDA exact eval.
