# Binary/source boundary audit

Verdict: PASS for the retained v13 receiver binary.

- The C translation unit implements only generic integer HPAC, probability,
  incremental causal-state, and RC64 algorithms.
- Learned weights, affine codes, residual values, group plans, and stream bytes
  enter through pointers populated from `archive.zip` by the Python binding.
- The native binary is retained twice with identical SHA-256
  `b791acf032c7f373beb329c3241323af04f8e939dd8c0195ac84ae908221779c`.
- `strings` on the retained binary found none of the archive, token, corrected
  logit, or corrected CDF SHA prefixes.
- A long-hex scan of `f26_hpac_native.c` and
  `experiments/ddm_f26q_f26_hpac_native.py` found no embedded payload hash or
  payload byte table.
- The archive remains exactly 186,269 bytes with SHA-256
  `f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de`.

The native lowering changes receiver execution only. It does not change archive
bytes, decoded tokens, rendered frames, scorer inputs, or score authority.
