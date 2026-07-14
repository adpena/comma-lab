# Codex harvest manifest — Bregman all-surfaces V9·CGauge — 2026-07-14

Pointer remains `0.1910828242 [contest-CPU Linux x86_64]`; local PR128
`0.1880443979880752` remains non-submission. No launch, score, evaluator, MPS,
or promotion claim is attached to this manifest.

## Commit disposition

Harvest only the exact owned new files below. Do not absorb sibling-owned
`vjp_fidelity.py`, surrogate policy, V9 config/provenance, basis, carrier, or
other dirty-worktree files. The Bregman files import the still-uncommitted
sibling canonical metric implementation, so a serializer must fail closed
rather than create a clean-HEAD-breaking partial commit unless that dependency
is first landed by its owner.

The exact serializer attempt returned `rc=6` on that dependency fence before
staging any file. The index remained empty and `HEAD` remained `927ef10723`.

## Owned new files and exact SHA-256

| SHA-256 | Path |
|---|---|
| `434a8e739ff71867b7de8e69d26af7517a45871ea25f364257e837ca51b599a1` | `.omx/research/bregman_v9_all_surfaces_DAG_FEED_20260714.md` |
| `bdc01ae586c4467b18ebf4deee206242426ddc51da3dc47d8a2b8fff6cab8481` | `.omx/research/bregman_v9_all_surfaces_binding_20260714.json` |
| `4364b73d9247a35a0c5653ecbc722079f4d2901f9b50ca79c405ed527af2ca4b` | `.omx/research/bregman_v9_all_surfaces_build_spec_20260714.md` |
| `12b82ca3f9809339746cc03b48a3237643861dec9e9baec19852a184fa7f358c` | `.omx/research/bregman_v9_all_surfaces_measurement_20260714.json` |
| `004c9dbc72e1082e82b0eaf0608173bf6f770c552a3e6cdf0c93391feed1b897` | `.omx/research/bregman_v9_all_surfaces_timing_advisory_20260714.json` |
| `a0024e8df1628a2692ee6cd0a763c16af7690679b322781c78cef2075cd1306b` | `.omx/research/codex_findings_bregman_v9_all_surfaces_20260714_codex.md` |
| `4b6b28636513c5f3eb390a19952a4a3ca672b55296b4002b0a65e0030215839a` | `.omx/research/codex_premise_falsification_bregman_dual_euclidean_20260714_codex.md` |
| `44ae04589e3f1032340ca890c2b250c7dfa680063ed0db1a7c1a40aeaefd3aeb` | `.omx/research/codex_review_bregman_v9_all_surfaces_round1_20260714.md` |
| `3acfc2a4df37d695c572e69c4614c0842b959df9ba555896a9dfc2c8b8bd220c` | `.omx/research/codex_session_summary_20260714_bregman_v9_all_surfaces_codex.md` |
| `daa0de11eeeb215b3fa651f5ed6fad450f2ddb431898497da506f1c76fa8052d` | `src/tac/canonical_equations/bregman_v9_surfaces_20260714.py` |
| `6d1428b67a4b8c0f008989d4b49a35370fadf21b109cb29a716f49582cc25547` | `src/tac/canonical_equations/tests/test_bregman_v9_surfaces_20260714.py` |
| `d9307ec64e5f84f0ad7b3e32ea7446e611590cad5bee91631b5f2e8f2b702cbd` | `src/tac/information_geometry/__init__.py` |
| `78aa17f65f35478adbf1d028a9887e397786933c9de5d1dec7c150fb6095fb4e` | `src/tac/information_geometry/bregman_v9_surfaces.py` |
| `6a4e83ad2d82a3c12a79d2493ab1469827ed27c43cf8f0a6dbf35789fb1228de` | `src/tac/tests/test_bregman_v9_surfaces.py` |
| `8c9fd6fe18de9e3df4ee9766e23698dd59921fd74e2f4e6168be8c369bfdd4e1` | `src/tac/tests/test_probe_bregman_v9_all_surfaces.py` |
| `27cf94b158acb45e18e3682d143fb23332c41bd49f424a9312c30cbdb1b72b13` | `src/tac/witness_dsl/bregman_geometry_policy.py` |
| `7703df170e0cdf55cef3e059df4db77798fd8a668c7fd74200fee3db3d81f20b` | `tools/probe_bregman_v9_all_surfaces.py` |

## Shared-file hunk harvest — do not stage whole files

These working-tree hashes include concurrent sibling changes. Harvest only the
Bregman/KL-reduction hunks and review them against the current base.

| Working-tree SHA-256 | Path | Bregman-owned seam |
|---|---|---|
| `4646ceae0fc1573c13004a4e8de0f2bfac318ea1a02aad7bcf5cd2cf4a28d419` | `src/tac/findings_lagrangian/info_gain.py` | sampled-KL fallback routes through strict extended estimator |
| `fbcb9f9760e4900aa949f8e37b941de33c90c2b2d78d1b85449a376a7417c033` | `src/tac/losses/u_die_kl.py` | spatial/generic KL reduction guard |
| `0f12b306d23a123db086ed19812cf9cadffe4be652a80dbb14f6513c7cd92902` | `src/tac/substrates/nscs02_downsampled_renderer/score_aware_loss.py` | explicit KL reduction custody |
| `f19ab9ed2df2eb957edb459a2fcd4bee3c0e357d367abc07b972f3ff8f8ba2d9` | `src/tac/training_curriculum/pause_distill_resume.py` | explicit KL reduction custody |
| `47fabf1fa63d447fb634dd1c1e43efc591878d63d15ed74442a5e5d8e4efdc4b` | `src/tac/dinov3_cooperative_receiver_anchor.py` | explicit KL reduction custody |
| `5d4e8f95596dc0525fbf388490a1c25b6305ef863d933a834178990175b00f04` | `src/tac/freezing/frozen_teacher_distillation.py` | explicit KL reduction custody |
| `592348c89444660b560240e64838dd98f82218b5bf7187c6868a87154462ced9` | `src/tac/freezing/tests/test_frozen_teacher_distillation.py` | resolution-invariance regression |
| `72d5b5aefb42712e0c50039d421a81134eb533575f1e32187bfc77ee97251362` | `src/tac/canonical_equations/deepmath_amortizing_argmax_laws_20260704.py` | correct `CE=H+KL` statement |

The append-only registry is concurrently active, so no whole-file hash is an
honest harvest authority. Harvest only lines 714–719, the six latest Bregman
equation events; every row
binds DAG SHA-256
`434a8e739ff71867b7de8e69d26af7517a45871ea25f364257e837ca51b599a1`.

## Unowned dependency custody at freeze

These actively edited paths are recorded only for dependency diagnosis, not as
content-addressed inclusion authority. Re-read their owner receipts after they
land; do not absorb them from this dirty worktree.

| Path | Owner status |
|---|---|
| `src/tac/scorer_surrogate/vjp_fidelity.py` | canonical metric owner; do not absorb |
| `src/tac/witness_dsl/surrogate_vjp_fidelity_policy.py` | canonical metric owner; do not absorb |
| `src/tac/witness_dsl/spec_v9_cgauge.py` | provenance owner hot file; do not absorb |
| `src/tac/witness_dsl/config_provenance.py` | provenance owner hot file; observed changing during final audit; do not absorb |
| `src/tac/witness_dsl/tests/test_config_provenance.py` | provenance owner hot test; do not absorb |

## Verification freeze

- Sealed receipt/binding validation: PASS.
- Owned Bregman/metric/equation/receipt/policy group: three clean runs of
  `71 passed`.
- KL consumer/shape group: `193 passed`.
- Ruff, format check, py_compile, and targeted diff check: PASS.
- End-to-end V9: `16 passed, 2 failed, 24 errors`, all rooted in the exclusive
  provenance owner's fail-closed scientific-declaration seal mismatch
  (`6cfa9845...` expected, `5c926130...` live) before Bregman assertions.
- Inbox consumed through arm `2026-07-14T14:23:10Z` and fleet
  `2026-07-14T15:56:40Z`; no stop directive.
