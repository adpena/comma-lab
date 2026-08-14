# Binary/source boundary audit

Verdict: PASS for the retained F26R optimized and scalar-twin receiver binaries.

- The C translation unit implements only generic integer HPAC, probability,
  incremental causal-state, and RC64 algorithms.
- Learned weights, affine codes, residual values, group plans, frame codes,
  context-convolution weights, conv-A deltas, and stream bytes enter through
  pointers populated from `archive.zip` by the Python binding.
- The optimized binary is retained twice at 69,424 B with identical SHA-256
  `1cf0e61b53d5b25a2b0cbb6adb47232921ebd442aa461cfcbb8db97d664a6aae`.
- The `F26_FORCE_SCALAR=1` twin is retained twice at 69,424 B with identical
  SHA-256 `64efe1e803aa0d22dbb0e3d02df5e7799a2e76b7ae4298311e78ab96cc86f4a8`.
- `strings` on both retained binaries found none of the archive, token,
  corrected-logit, or corrected-CDF SHA prefixes.
- A long-hex scan of `f26_hpac_native.c` and
  `experiments/ddm_f26q_f26_hpac_native.py` found no embedded payload hash or
  payload byte table.
- The archive remains exactly 186,269 bytes with SHA-256
  `f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de`.

The optimized, repeat, and scalar-twin n600 fields all have token SHA
`9ba2e52b...`; their corrected-logit SHA, CDF SHA, and RC64 bit position also
match exactly. The lowering changes receiver execution only. It does not change
archive bytes, decoded tokens, rendered frames, scorer inputs, or score authority.
