# PR92 QMA9 Rust Lowering Worker - 2026-05-04

## Scope

Local-only Rust lowering/hardening for PR91/PR94 HPAC/HPM1 and PR92
range-mask surfaces. No remote GPU, training, exact eval, scorer invocation, or
dispatch was performed. No score claim is made here.

## Inputs Inspected

- PR92 public intake: `experiments/results/public_pr92_intake_20260504_codex/`
  - PR number: `92`
  - Archive bytes/SHA-256: `236516`,
    `f0dedeb7ad3c019ab3f4e2dea317f3192408e47a573897adb208030709d01490`
  - Member `x` bytes/SHA-256: `235952`,
    `89f14d331063125c88db0b4e3e51a92f21d2edc175a64d5c9cb6f873130763d8`
  - Member `a` bytes/SHA-256: `386`,
    `5422d47b4092e7304649cc49f4d4c8c7efa9c3d5a4fc7d39ab63cf2518e0897e`
  - `range_mask_codec.cpp` bytes/SHA-256: `53975`,
    `94cd1a86111fb6d34b6e12d37c624bd5938df0fbc6c4c24c8d40c5a83fcb016b`
  - `inflate.py` bytes/SHA-256: `102838`,
    `c04cba8431f076838031f013aa005cdb7f530133d5d1856bfe0f02def3f11261`
- PR91 HPM1 intake: `experiments/results/public_pr91_intake_20260504_codex/`
  - Relevant contract surface remains `src/tac/pr91_hpm1_codec.py` and
    `runtime-rs/crates/hpac-codec`.
  - Known HPM1 mask segment bytes/SHA-256 from current Python contract:
    `145087`,
    `a4ed57ff0af1d8c914f004de165aeead50ec8dd61e99b0afdfbfa2d1e7fd9fcc`
  - Known HPM1 tokens SHA-256:
    `541016d83852a5bb3e0738caa3b44d7b2b0f7372f1841085cf9554f039c6cf6b`
- PR94 HPAC probe: `experiments/results/public_pr94_hpac_contract_probe_20260504_codex/`
  - Files inspected: `pr94_pr86_hpac.py`, `pr94_hpac_inflate.py`
  - No archive fixture was present in this probe directory.

## Rust Lowering Added

File: `runtime-rs/crates/qma-codec/src/lib.rs`

Added a deterministic, fail-closed parser for the PR92/PR91-style compact
bundle v5 micro header:

- 24-byte header of eight 24-bit little-endian lengths.
- Fixed `bias=223` and `region=273` byte tails.
- Nonempty `randmulti` tail requirement.
- Native mask-slice boundary helper before QMA9 decode.

The new Rust fixture pins PR92's concrete anatomy:

- Header bytes: `236d02f2de00cf0500780500e200006a00009500009a0000`
- Mask bytes: `159011`
- QMA9 header: `(frames=600, width=512, height=384, bitstream_bytes=158991)`
- Decoded mask byte contract: `117964800`
- Model/pose/post/shift/frac/frac2/frac3 bytes:
  `57074`, `1487`, `1400`, `226`, `106`, `149`, `154`
- Randmulti tail bytes: `15825`

This is archive anatomy and bitstream-boundary hardening only. It does not
decode PR92's full archive, run the scorer, or promote a result.

## Verification

```bash
cargo fmt -p qma-codec
cargo test -p qma-codec
```

Result: `8 passed` across qma-codec unit and CLI tests.

## Next Highest-EV Native Lowering Target

Lower the PR92/QMA9 adaptive binary arithmetic encoder into Rust next, not just
the decoder. The current C++ has macro-tunable priors and encoder paths, while
Rust already owns decode and now owns compact-bundle slicing. A native
encode-decode parity fixture over the real PR92 mask slice SHA
`4b9d93fedb37a9d6fd435054cc33e216d703818b3ac94f4616c89969a4e0d179`
would make future QMA prior sweeps and HPAC/HPM1 comparisons deterministic
without recompiling ad hoc C++ in the inflate path.
