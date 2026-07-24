# Codex findings — DDM E4 Brotli declared dependency — 2026-07-24

`research_only=true` · `score_claim=false` ·
`[macOS-CPU upstream frozen-harness advisory]` ·
pointer `0.1910828242 [contest-CPU]` unchanged · MAIN landing review required.

## Verdict

**PASS the bounded E4 rate-recovery and dual-runtime contract. BLOCK
promotion because no contest-CPU/CUDA replay or pointer mutation was
authorized.**

Brotli 1.2.0 Q11 is the primary real coder and declared runtime dependency
number two after Torch. The only fallback trigger is `ImportError` while
importing Brotli; that path uses the byte-exact E3 raw-LZMA1 contract. A Brotli
coder exception after a successful import propagates and is never masked.

The selected E4 archive is 344,203 bytes, SHA-256
`d1a1a426a7b1a87287738ed6ced3d1a2298bb68efe5866c6fad12716f1d6e372`.
Relative to E3's 439,303-byte archive, the exact recovery is 95,100 bytes or
`0.0633231864419185` score units under the named rate metric
`25*archive_bytes/37,545,489`. The prior 95,837-byte projection is corrected
down by 737 bytes: 484 bytes of canonical typed-stream custody plus the
previously measured 253-byte E4 dependency/coder-manifest correction.

There is no distortion-side trade in this lane. Both lossless coders decode to
the same raw witness bytes.

## Exact byte accounting

E3 raw-LZMA1 untagged to selected E4 Brotli tagged:

| Byte home | E3 before | E4 after | Delta bytes | Delta score units |
|---|---:|---:|---:|---:|
| `manifest.json` | 9,858 | 10,286 | +428 | +0.0002849876319362894 |
| `base/chart.ddb` | 17,825 | 18,469 | +644 | +0.0004288131658106784 |
| `semantic/composed.dds` | 411,274 | 315,102 | -96,172 | -0.06403698723966547 |
| ZIP headers + central directory | 346 | 346 | 0 | 0 |
| **archive total** | **439,303** | **344,203** | **-95,100** | **-0.0633231864419185** |
| `typed_stream_tag_overhead` | 0 | 484 | +484 | +0.00032227573331113096 |

The typed-stream row decomposes `manifest.json`; it is not an additional byte
home to sum into the archive total.

The like-for-like tagged fallback-to-primary coder A/B is 439,787 to 344,203
bytes, exactly -95,584 bytes or -0.063645462175230 score units. Its section
deltas are -56 manifest bytes, +644 chart bytes, -96,172 semantic bytes, and
zero container bytes. The semantic stream therefore owns all net coder
recovery; Brotli is 644 bytes worse on the chart.

## Typed-stream decision

MAIN directive `2026-07-24T03:33:42Z` offered two alternatives. E4 selects:

**`KEEP_AND_VERSION_BUMP_E4_MANIFEST`.**

The exporter uses the canonical
`tac.optimization.ddm_min_description_contract.TypedStreamTag`,
`StreamType`, and `LayerHome`; it defines no parallel enum. Both tags live in
the counted manifest and reconcile exactly to their section byte counts:

- `base/chart.ddb`: `FIBER`, `L2_chart`, 18,469 counted bytes.
- `semantic/composed.dds`: `SKELETON`, `L1_program`, 315,102 counted bytes.

Exact A/B stripping removes 484 bytes from either coder's archive and
reproduces each stored untagged control archive byte-for-byte. The selected
receiver nonetheless refuses missing or malformed tags under the versioned
`ddm_e4_runtime_archive.v1` schema. The 0.000322276 score-unit cost is retained
because fail-closed byte-home custody is the stronger contract.

## Dual locked upstream receipts

Both runs used the same frozen SSD upstream clone. `evaluate.py` SHA-256 is
`7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b`;
`evaluate.sh` SHA-256 is
`9612284ce6e9585aefcf636f3027808a56160ffd572edffdf4b8622a65fac917`.

| Runtime arm | Brotli installed | Archive bytes | Raw SHA-256 | d_seg | d_pose | Rounded advisory score | Wall clock |
|---|---:|---:|---|---:|---:|---:|---:|
| primary Brotli Q11 | yes, 1.2.0 | 344,203 | `4c553508b0bf92ccdc137e215799ae30a346b58e0617e5156441a7929302b4f1` | 0.02861482 | 147.49104309 | 41.50 | 1,220.568330 s |
| E3 raw-LZMA1 fallback | no (`find_spec("brotli") is None`) | 439,787 | `4c553508b0bf92ccdc137e215799ae30a346b58e0617e5156441a7929302b4f1` | 0.02861482 | 147.49104309 | 41.56 | 1,299.967835 s |

Both receipts have exit code zero, empty failure reasons, exact raw byte count
3,662,409,600, 38 preserved receiver stages, and wall clock below 1,800
seconds. The rounded score change is therefore purely the counted rate term.
It is not a new contest score or pointer.

## #417 counted-section consumption proof

The verifier decodes both real sections from the selected Brotli archive and
the fallback archive and proves exact raw equality:

- chart: 1,404,000 bytes, SHA-256
  `5aa4cb7bb1e05b8dd9d5191adac3ecb111085bc83bdc5cc91d04eb8a9a400e51`;
- semantic: 117,964,800 bytes, SHA-256
  `15f14ece3276c5e7a6a5ee4a328edeb377271c8d016d28d932575001fba079ff`.

For each selected Brotli section, flipping the terminal coded byte causes the
receiver to refuse the section. This is the #417 counted-to-output no-op
detector: neither counted section is ignored, and corruption cannot survive
as a silent no-op.

## Directive disposition

| Directive | Disposition |
|---|---|
| Brotli runtime dependency number two via manifest/bootstrap | PASS: primary manifest declares `["torch","brotli"]`; `inflate.sh` selects the locked declared environment |
| Brotli Q11 primary | PASS |
| exact E3 raw-LZMA1 fallback only on `ImportError` | PASS; real import-failure generation and Brotli-absent full harness |
| do not mask coder errors | PASS regression |
| dual locked `evaluate.sh` | PASS/PASS with exact raw and metric identity |
| per-section real composed-payload measurement | PASS; table above and machine receipt |
| correct -95,837-byte projection | PASS: actual -95,100 bytes, +737-byte correction |
| #417 section-consumption proof | PASS |
| canonical `TypedStreamTag` and named metric law | PASS |
| local n600 only; no paid/remote/pointer actuation | PASS |
| fixed Torch receiver threads | PASS: four |
| MAIN landing review | **required and pending** |

## Triality

- DSL: `DDME4RuntimeExporterConfigV1`,
  `DDME4UpstreamHarnessConfigV1`, and `ddm_e4_runtime_archive.v1`.
- DAG: E3 counted payload → selected real coder → typed counted sections →
  declared runtime dependency environment → preserved receiver stages → exact
  raw → locked frozen scorers.
- Equations: `real_coder_archive_bytes_contest_units_v1` with
  `25*B/37,545,489`; `ddm_min_description_contract` typed-stream byte homes;
  `ddm_runtime_export_identity_receiver_closed_v1`.

Machine receipts:

- `.omx/research/ddm_e4_brotli_declared_dep_20260724/ddm_e4_brotli_rate_recovery_receipt.json`
- `.omx/research/ddm_e4_brotli_declared_dep_20260724/brotli_tagged/ddm_e4_runtime_export_receipt.json`
- `.omx/research/ddm_e4_brotli_declared_dep_20260724/brotli_tagged/ddm_e4_brotli_upstream_harness_receipt.json`
- `.omx/research/ddm_e4_brotli_declared_dep_20260724/lzma1_fallback_tagged/ddm_e4_runtime_export_receipt.json`
- `.omx/research/ddm_e4_brotli_declared_dep_20260724/lzma1_fallback_tagged/ddm_e4_lzma1_fallback_upstream_harness_receipt.json`

## STORES CONSULTED

- `CLAUDE.md` and `AGENTS.md`, fully.
- `.omx/research/OPERATING_MANUAL.md`.
- E3 findings, DAG-FEED, runtime/export/upstream receipts, and coder table.
- Canonical typed-stream landing `f349c6eca3` and
  `src/tac/optimization/ddm_min_description_contract.py`.
- Canonical lane registry, subagent-progress ledger, per-arm inbox, and
  broadcast inbox.
- Exact E4 export, rate, receiver, and dual upstream receipts listed above.

## Independent review record

Round 1 disposition: **PASS bounded implementation and advisory evidence;
BLOCK promotion.**

The review focus for MAIN is mandatory:

1. Reconcile this branch against already-landed typed-stream commit
   `f349c6eca3`, preserving its full canonical headline-contract additions
   while retaining E4's Brotli-specific code.
2. Re-derive that only an import-time `ImportError` selects raw LZMA1 and that
   every post-import Brotli error propagates.
3. Verify the strict E4 manifest/dependency/coder cross-check and tag-absent
   refusal in the emitted standalone receiver.
4. Recompute the 95,100-byte E3-to-selected recovery and 95,584-byte
   coder-only recovery from member homes, including the 484-byte tag row.
5. Keep both locked receipts advisory-only and do not move the
   `0.1910828242 [contest-CPU]` pointer.

Three consecutive clean passes:

1. Ruff format/check, Python compile, 25 focused contract/runtime tests, nine
   canonical JSON checks, two emitted-runtime source identities, exact rate
   verifier replay, dual raw/metric identity, whitespace check, and full diff
   review: PASS.
2. Loaded each emitted `inflate.py` independently of the exporter in its real
   environment; validated both exact manifests, decoded both real sections,
   proved `brotli_spec=None` in the fallback environment, refused a tag-absent
   manifest, re-derived exact rational score arithmetic, and reran 25 tests:
   PASS.
3. Rebuilt both archives deterministically from stored members, proved the
   fallback chart/semantic bytes are the exact E3 section bytes, revalidated
   both locked receipts and all 38+38 preserved stages, rehashed the three
   terminal receipts, checked the scoped lane-registry invariants, and reran
   lint/compile/25 tests: PASS.

The review tracker records 104 reviewed entities in each of
`ddm_e4_round1`, `ddm_e4_round2`, and `ddm_e4_round3`. A separate global
`lane_maturity.py validate` reports 110 historical missing-evidence paths in
this isolated worktree; none belongs to this lane. The scoped E4 lane is
unique, L2/phase 4: implementation, real-archive empirical, strict-preflight,
and three-clean-review gates are true. Contest-CPU, contest-CUDA, memory, and
deploy-runbook gates remain correctly false.

MAIN landing review is required.
