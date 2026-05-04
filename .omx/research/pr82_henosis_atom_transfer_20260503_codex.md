# PR82 Henosis Atom Transfer Local Screen - 2026-05-03

No remote GPU dispatch was performed.

## QRM1 Runtime Greenup Addendum

Updated robust runtime status:

- `submissions/robust_current/apply_qzs3_postprocess.py` now decodes
  Brotli(`QRM1`) from the `QPS1` randmulti stream.
- Runtime QRM1 reads the carried `u16 replay_group_id` and applies the
  reviewed PR82 replay spec table.
- Generic PR82 randmulti groups and raw-frame-only PR82 special branches now
  have focused runtime parity tests.
- Mask-dependent PR82 special branches still fail closed with explicit
  diagnostics. They are not dispatchable until ported or excluded by a
  candidate-specific parity proof.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_qzs3_postprocess_qrm1_runtime.py src/tac/tests/test_build_pr82_henosis_atom_transfer_candidates.py -q`
  -> `12 passed`
- broader focused replay/stack surface -> `45 passed`

The refreshed PR81+PR82 stack manifests now treat QRM1 decode itself as
implemented. The current blocker is raw-output parity plus mask-dependent
special semantics, not missing QRM1 parsing.

## PR81 QMA9 + PR82 Henosis Stack Addendum

Updated artifacts:

- `experiments/build_pr81_pr82_henosis_stack_candidate.py`
- `experiments/results/pr81_pr82_henosis_stack_20260503_codex/candidate_summary.json`
- `experiments/results/pr81_pr82_henosis_stack_20260503_codex/DESIGN_NOTE.md`

No remote GPU dispatch was performed.

This pass targeted the high-leverage PR81/QMA9 bytes plus PR82/Henosis
component-quality hypothesis.  Inputs:

- PR81/QMA9 archive: `215960` bytes, SHA-256
  `cd01378a52688fe00ee1bfb898c67695aed89a7b3ded602b597eb7fc3031d7fc`
- PR82/Henosis archive: `296789` bytes, SHA-256
  `a0e07c360223c1dd3d3b92263225d38d542e218e83d095ad9b91bf872f94c6e4`
- PR82 exact T4 components used for planning-only lower bounds:
  SegNet `0.00057185`, PoseNet `0.0001894`

Static lower bounds if PR82 components carried unchanged:

- ideal PR81 bytes: `215960` bytes, expected score `0.244504009810242`
- compact PR81-mask + PR82 model/pose only: `217696` bytes, expected score
  `0.2456599409528621`
- compact PR81-mask + PR82 model/pose/post controls: `220227` bytes,
  expected score `0.2473452299632143`
- compact PR81-mask + full PR82 post/randmulti tail: `236328` bytes,
  expected score `0.25806622496743437`

Deterministic local sidecar candidates emitted:

- `pr81_qma9_pr82_qps1_controls_all600`: `218621` bytes, `qpost.bin`
  `2567` bytes, planning score if PR82 components carried
  `0.24627586048450012`
- `pr81_qma9_pr82_qps1_nm2_generic_randmulti`: `223392` bytes,
  `qpost.bin` `7338` bytes, planning score if PR82 components carried
  `0.249452673549846`
- `pr81_qma9_pr82_qps1_qrm1_all072_randmulti`: `232580` bytes,
  `qpost.bin` `16526` bytes, planning score if PR82 components carried
  `0.2555705856111325`
- `pr81_qma9_pr82_qps1_controls_qrm1_all072`: `235111` bytes,
  `qpost.bin` `19057` bytes, planning score if PR82 components carried
  `0.2572558746214847`

Smallest current-helper sidecar is controls-only, but the highest-EV quality
transfer candidate is `pr81_qma9_pr82_qps1_controls_qrm1_all072`: it carries
all PR82 control streams plus exact sparse-row `QRM1` custody for all 72
randmulti groups while staying `61749` bytes below wholesale PR82.

Dispatch remains fail-closed.  Unsupported semantics before any exact T4 run:

- `robust_current/unpack_renderer_payload.py` does not parse the PR81 public
  QMA9 `p` payload;
- `submissions/robust_current/apply_qzs3_postprocess.py` does not decode
  `QRM1`;
- the PR82 frame-1 randmulti special branches are not ported to robust runtime;
- no local raw-output parity/delta proof is attached for these exact archive
  SHAs;
- any future exact eval requires a dispatch lane claim.

## Artifact

- Tool: `experiments/build_pr82_henosis_atom_transfer_candidates.py`
- Summary: `experiments/results/pr82_henosis_atom_transfer_20260503_codex/candidate_summary.json`
- Source PR82 archive: `experiments/results/public_pr82_henosis_frontier_intake_20260503_codex/archive.zip`
- Source PR82 SHA-256: `a0e07c360223c1dd3d3b92263225d38d542e218e83d095ad9b91bf872f94c6e4`
- Transfer base: `experiments/results/pr79_action_dictionary_repack_v2_20260503_codex/pr79_s2_fixed_adaptive_actions/archive.zip`
- Transfer base SHA-256: `5740aca7e255b00093154eb1823b5b6207d8795f8eb287d35758c4cda438ec68`

## Static Rate Finding

PR82 remains a rate-negative wholesale transplant against PR79/S2:

- PR82 bytes: `296789`
- PR79/S2 bytes: `277321`
- Delta: `+19468` bytes
- Formula rate penalty: `0.012962942099382433`

The value of PR82 is therefore atom signal, not wholesale rate.

## Local Candidates

The tool emitted deterministic local archives:

- `pr82_qpost_top008_on_source`: `277776` bytes, `+455` vs PR79/S2, 58 active qpost atoms
- `pr82_qpost_top016_on_source`: `277886` bytes, `+565` vs PR79/S2, 115 active qpost atoms
- `pr82_qpost_top032_on_source`: `278007` bytes, `+686` vs PR79/S2, 220 active qpost atoms
- `pr82_pose_velocity_top008_on_source`: `277326` bytes, `+5` vs PR79/S2, 8 changed QP1 velocity words
- `pr82_pose_velocity_top016_on_source`: `277325` bytes, `+4` vs PR79/S2, 16 changed QP1 velocity words
- `pr82_pose_velocity_top032_on_source`: `277322` bytes, `+1` vs PR79/S2, 32 changed QP1 velocity words

Evidence grade: `empirical_local_archive_build_and_atom_accounting`.

## Dispatch Gate

All candidates are fail-closed for dispatch:

- no raw-output delta proof was attached in this local screen;
- no component-trace proxy clears the PR79/S2 rate break-even;
- nearby PR65/PR75 qpost and pose transfer families already have exact T4 negative evidence;
- any future exact eval requires a fresh lane claim before dispatch.

## Adversarial Profile

PR82 randmulti is not directly transferable into the current `QPS1` helper:
the replay declares 72 headerless randmulti groups, while the robust helper
supports the older 16-group headerless shape. The initial builder therefore
omitted randmulti from qpost candidates and recorded it as runtime-incompatible;
the addendum below narrows that to a compatible generic subset plus replay-only
blocked groups.

PR82 P1D1 dimension 2 is also not directly transferable onto PR79/S2 QP1,
which carries only the velocity column. Non-velocity pose transfer needs a
reviewed PVR1/QP2 path before it can become dispatchable.

## Low-Level Randmulti Deconstruction Addendum

Updated artifact:
`experiments/results/pr82_henosis_atom_transfer_20260503_codex/pr82_randmulti_lowlevel_profile.json`.

The PR82 `x` member still closes as a v5 micro-header compact bundle:
eight 24-bit little-endian lengths for `mask`, `model`, `pose`, `post`,
`shift`, `frac`, `frac2`, and `frac3`, followed by fixed replay-tail lengths
`bias=223` and `region=273`, then the residual `randmulti` Brotli tail.
The PR82 archive remains `296789` bytes with SHA-256
`a0e07c360223c1dd3d3b92263225d38d542e218e83d095ad9b91bf872f94c6e4`.

The randmulti tail decodes to `27105` bytes and 72 replay-spec groups.  It is
not uniformly incompatible: 35 groups are exactly representable by the current
robust `QPS1` helper through Brotli(`NM2`) dense groups, because they use the
generic frame-0 nearest-neighbor random-pattern path and fit the `NM2` u8
`height,width,amplitude,scount` fields.  Those 35 groups contain 5738 nonzero
choices.  The remaining 37 groups are replay-only under the current helper:
large generic random-pattern groups exceed `NM2`'s u8 dimension fields, and
the high-value f2 tile/boundary/class-bias groups use PR82 replay special
branches that target frame 1 rather than the helper's generic frame-0 path.

New deterministic byte screens were emitted for the compatible subset:

- `pr82_randmulti_generic_top001_on_source`: `278957` bytes, `+1636` vs
  PR79/S2, 1194 active choices, `qpost.bin` 1542 bytes.
- `pr82_randmulti_generic_top004_on_source`: `279906` bytes, `+2585` vs
  PR79/S2, 2153 active choices, `qpost.bin` 2491 bytes.
- `pr82_randmulti_generic_top008_on_source`: `280800` bytes, `+3479` vs
  PR79/S2, 2926 active choices, `qpost.bin` 3385 bytes.

These screens are runtime-compatible but dispatch-blocked: no raw-output delta
proof or component trace clears the rate penalty.  They are useful primarily
as charged atom/economics screens and as a proof that the parser can reproduce
PR82 randmulti below the opaque tail level.

Compatible charged representation now implemented for local screening:
`QPS1` randmulti stream containing Brotli(`NM2` dense groups) over the exact
generic PR82 group rows.  Proposed next representation for the full 72-group
family: `QRM1`, a sparse group-id stream carrying `u16 replay_group_id` plus
sparse rows, with runtime table lookup for both generic large-dimension
patterns and PR82's f2 special-case tile/boundary/class-bias semantics.  This
would avoid dense `NM2` expansion and make the replay-only groups chargeable
without relying on headerless parser divergence.

## QRM1 Native Contract Addendum

Updated artifacts:

- `experiments/results/pr82_henosis_atom_transfer_20260503_codex/candidate_summary.json`
- `experiments/results/pr82_henosis_atom_transfer_20260503_codex/pr82_randmulti_lowlevel_profile.json`
- `experiments/results/pr82_henosis_atom_transfer_20260503_codex/pr82_randmulti_qrm1_all072_on_source/manifest.json`

Implemented local deterministic `QRM1` encode/decode/profile support.  The
contract is Brotli(`QRM1` + `u16 group_count` + repeated `u16 replay_group_id`
and sparse PR82 rows).  The replay group id anchors the runtime semantics in
the reviewed PR82 table instead of trying to squeeze large or special groups
into `NM2`'s u8 dimensions.

Local parity profile:

- decoded groups: `72`
- exact group-row parity: `true`
- missing/extra/spec/row mismatch ids: none
- nonzero choices represented: `13496`
- generic groups: `62`
- replay-special groups: `10`
- source PR82 randmulti tail: `16101` Brotli bytes, `27105` decoded bytes
- QRM1 randmulti stream: `16490` Brotli bytes, `27255` decoded bytes
- QRM1 delta vs PR82 source tail: `+389` Brotli bytes

New deterministic local candidate:

- `pr82_randmulti_qrm1_all072_on_source`: `293941` bytes, `+16620` vs
  PR79/S2, `qpost.bin` `16526` bytes, randmulti stream `16490` bytes,
  archive SHA-256
  `a86166e634788eea4e2ea4acd06e7650a81f501fff2d92ca3630ce7b1af8e56d`.

Dispatch remains fail-closed.  The local `QRM1` contract proves custody and
sparse choice-row parity for all 72 PR82 groups, but
`submissions/robust_current/apply_qzs3_postprocess.py` has not been changed in
this scoped pass.  Remaining runtime work before any exact-eval consideration:

- read Brotli(`QRM1`) from the `QPS1` randmulti stream;
- look up PR82 replay specs by carried group id;
- apply generic large-dimension groups through the existing deterministic
  frame-0 random-pattern path;
- port the PR82 frame-1 special branches for tile, boundary, class-conditioned,
  boundary/class, width-2/width-3 boundary, all-channel, and global RGB-bias
  semantics;
- attach a local raw-output parity/delta proof before any dispatch claim.

No remote GPU dispatch was performed.
