# Codex findings — genuine curvelet/shearlet build + measure — 2026-07-14

`FEED-502` · `research_only=false` · `pointer_moved=false` · `operator_go_containment=true`

## Receipt-bound authority

The fresh COMPLETE receiver and mask receipts were re-opened through
`tac.canonical_equations.optimal_basis_selection_20260714:ingest_completed_advisory_genuine_receipts`.
The resulting sidecar is
`.omx/research/genuine_curvelet_shearlet_canonical_advisory_ingest_v2_polar_frequency_wedge_20260714.json`.

| surface | receipt SHA-256 | COMPLETE progress SHA-256 | authority |
|---|---|---|---|
| receiver, `REALIZED_THROUGH_R_SAVED_OFF_RGB_NTERM`, n=600 | `031a1569c600bf5d1a3551a4da668db67bb80380d0678f964ca5476e0b293c33` | `ae281589cadf57c3fbbbcbe50fe3d1c4f3f10d8706abfd67e403369e0c47daec` | macOS-CPU advisory, realized-through-R CPU-SegNet, non-promotable |
| mask proxy, `MASK_SPACE_PROXY_NOT_THROUGH_R_NOT_SCORE`, n=600 | `72c0c1f736e68267d346885792901d64f0c7c3ff83952e4a7a41831330587aa0` | `d513c3e256b8986eef003f807f1b66060edc60d61646af41e369cbe9880ac23b` | proxy only; receiver inverse unavailable |

Ingest status is `MEASURED_ADVISORY_NOT_BYTE_CLOSED`, with
`archive_byte_closed=false` and `selection_eligible=false`. The metric is exactly
`argmax_native_vjp_fidelity_v1`; no score, promotion, or family-kill claim is made.

## Structural scope

Structural proof v2 is COMPLETE at
`.omx/research/genuine_curvelet_shearlet_structural_proof_v2_polar_frequency_wedge_20260714.json`
(`sha256=677a2252c43c1272ec0e2e83d65ce1b82d23b8ddb089d73a111a5f0b26d46d25`). Claims are
finite discrete polar-frequency wedge / finite localized curvelet and compact-shearlet
truncations only. Continuum tight-frame, completeness, approximation-rate, and
continuum PoU claims are **NOT_CLAIMED**. The measured fourth arm is a fixed
`4 Q1 + 38 polar + 38 curvelet` mix; it is not the requested decoder-boundary PoU.
Literal PoU remains `BLOCKED`; the mask cannot be treated as a decoder inverse.

Round-1 artifacts/ranks are explicitly `INVALID` and are not inputs to this receipt.
The receiver parameter interpretation is the single post-checkpoint
`int8_dequant_params` round trip; the raw fp32 shadow is not receiver authority.
Receiver and mask git SHAs differ due concurrent main movement; all seven runtime/code
source hashes match and strict live rehash passed.

## Fresh measured curves

All four arms have equal coefficient scalar values and support-index counts
`13,695 / 27,390 / 54,780 / 109,559`; this does not assert equal parameters or archive
bytes. Values below are copied from the receipt surfaces (receiver `d_seg` through R;
mask `d_mask_argmax` is a separate proxy).

| arm | receiver d_seg (.125, .25, .5, 1.0) | mask proxy (.125, .25, .5, 1.0) |
|---|---|---|
| polar directional Fourier | 0.6277346038818359, 0.6005332777235243, 0.47180653042263454, **0.40972231547037763** | 0.05273525661892361, 0.048664686414930554, 0.04872350904676649, 0.04866945054796007 |
| compact shearlet | 0.5654775746663412, 0.4538553195529514, 0.33307122972276476, **0.42886043124728734** | 0.11530280219184028, 0.11605059305826823, 0.11053274366590712, 0.11024330139160156 |
| windowed curvelet | 0.5078373294406467, 0.5045310295952691, 0.5048346201578776, **0.5048239559597439** | 0.24489429897732204, 0.23919029235839845, 0.2391814253065321, 0.23918173048231336 |
| fixed 4 Q1 + 38 polar + 38 curvelet | 0.49349054124620223, 0.4831833224826389, 0.49360439724392363, **0.5303014119466146** | 0.052096227010091145, 0.041628104315863716, 0.041714536878797746, 0.041684273613823784 |

Full receiver rank at budget 1.0 is **Fourier 0.40972231547037763 < compact shearlet
0.42886043124728734 < windowed curvelet 0.5048239559597439 < fixed mix
0.5303014119466146**. This is a formulation-level advisory ordering only. The mask
proxy ordering is separate and cannot override it; its disagreement is evidence for
the missing receiver inverse, not a score or basis verdict.

## Triality / remaining custody

- DAG leg: this findings memo plus the canonical sidecar and fresh receipt hashes.
- DSL leg: one typed basis lever remains the intended v9·CGauge integration point; no
  direct argv or standalone promotion is authorized.
- Equation leg: `optimal_basis_equal_budget_through_r_v1` consumes the sidecar. The
  live strict-ingest source SHA is
  `3eb00fba7f68cc9b37be5cbf48951d828c8af50b2eb006228800c5182fa3bcc0`; registry-only
  refinement is recorded by anchor row 721 (`9718486df60569cc9906fc114559a9ed9dc919a11561535658f594858642227b`)
  and domain row 722 (`93b309254e296aea79f892d74508538e19ce6cce5f31d4d07063b90cbc86041b`),
  registry SHA `4c9de561729707af245fcdb0d39c0aac9f3973dc357813f9f2beabf85ee9115d`.
  Anchor `genuine_polar_curvelet_compact_shearlet_n600_advisory_20260714` has residual
  `0.09510164048936631` and `selection_eligible=false`.
- Probe outcome: `.omx/state/probe_outcomes.jsonl` line 643, probe
  `genuine_curvelet_shearlet_equal_budget_nterm_n600_20260714`, verdict `PARTIAL`
  advisory, row SHA `ed118c0e7f1b355b1fdd4efb88c53c9662ae98b8b7baa030d59d744c8c412ae3`,
  whole-ledger SHA `871e4a5929c1b32e685dbc5ae11fd24e0574229b47a7f5934cbb7b04e9b064fb`.
- Gram status is `NO-VERDICT_DATA_CUSTODY`; no Fisher/Hessian Gram was measured.
- Pointer remains unchanged at `0.1910828242 [contest-CPU Linux x86_64]`.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`; project memory; lane/subagent registries;
live per-arm and fleet inboxes; the v2 structural proof; fresh receiver/mask receipts
and COMPLETE progress; the ingest callable; the typed equation source and registry;
the finalized fresh-start ticket (`sha256=41ace6c31a476ceb2f7f3160023362b034781fc26c935d008daa82883b3febb0`);
and the finalized partition-of-unity blocker (`sha256=76eaf17b0e2f7db56e9d4e9f81b0fa86ec090f7363811e59b2de6cb323e5a45a`).

Pointer delta: exactly zero.
