# F26 native corrector — gb1 20-family generation (git mirror of the SHIPPED lineage)

**Why this directory exists.** The repo-canonical pair one level up
(`../f26_corrector_native.c` + `../native_free_corrector.py`) is the ORIGINAL 13-family
ddm_rr8 port, kept as-is because `src/tac/tests/test_ddm_rr8_native_corrector.py` binds its
parity leg to a jg5-era base tree at that config. The SHIPPED lineage moved on twice without
a git mirror: fx5 added six members (19 families, dx2 pointer runtime) and ddm_gb1 added
`groupbin8_surprise` (20 families). Per the SSD-is-for-artifacts-only law, code must live in
git — this directory is the 20-family generation's git home, byte-identical to the fire tree
that produced the gb1 T4 candidate.

**Provenance (receipt-bound):**
- Source of truth: `/Volumes/APDataStore/pact/ddm_gb1_groupbin8_conditioning/runtime_fire_v1/runtime/`
  (content-only runtime digest `7a4b57935fb7fdf310729575a88094089fd4bf7bc5e0b134bb32b6f9675deecd`,
  sealed in `SEAL_fire_gb1_groupbin8.json`, seal sha `9a31811d71af3dcf…`).
- Port base: the SHIPPED 19-family fx5 C in the dx2 pointer runtime
  (`/Volumes/APDataStore/pact/ddm_dx2/r7/candidate_runtime_dx2/runtime/`) — NOT the
  13-family repo original. Delta vs that base = exactly the gb1 patch set (GROUP_BINS 8 ·
  N_FAMILIES 20 · RULE_GROUPBIN8_SURPRISE enum/table rows · `family_rule_index` argument +
  case · the causal `groupbin8 = (((x % 64) + 2*(y % 64)) * 8) / 190` at the caller).
- **Proof the port is exact:** full-payload decode via `inflate.sh` (native corrector
  engaged, fallback message absent) produced `0.raw` sha
  `7246a4ff8f79b03aa…` — byte-identical to the Python-corrector decode AND to the dx2
  pointer's retained macOS raw. 384 s wall vs ~24 min Python.
- Compile contract unchanged from rr8: `cc -O3 -std=c11 -shared -fPIC -ffp-contract=off
  -fno-fast-math … -lm` — `-ffp-contract=off` is load-bearing (FMA contraction changes the
  emitted probabilities and desynchronises the arithmetic decoder).

**Config gate.** `native_free_corrector.py` here carries the 20-family
`EXPECTED_SHIPPED_CONFIG` (families tuple + `GROUP_BINS: 8`); `assert_config_matches()`
refuses any runtime whose Python config disagrees with what this C compiled in. Do NOT pair
this generation with a 13- or 19-family runtime tree.

Verdict memo: `.omx/research/ddm_gb1_groupbin8_verdict_20260824.md`.
