# Task #574 — xi-keyed temporal delta coder

## Outcome

The exact n64 and n600 measurements reject this formulation. The planar-3 `(rho_z, rho_x, omega_y)` predictor derived from a counted composed full screw is bit-exact, but it makes the settled coherent-slot ground-frame Lane description larger:

| pairs | settled LBND2 terminal | identity + xi-context control | planar-3 xi predictor |
|---:|---:|---:|---:|
| 64 | 3,617 B | 5,194 B (+1,577) | 5,331 B (+1,714) |
| 600 | 35,393 B | 42,413 B (+7,020) | 43,901 B (+8,508) |

Verdict: **MEASURED_FORMULATION_NEGATIVE** for the planar-3 projection on the already ground-canonicalized coherent-slot LBND chart in this S4 corpus. This does not close full-6D prediction, non-Lane temporal coding, uncanonicalized charts, or the xi family.

Source-object scope is a separate hard boundary. The 451,191 B S4 artifact is a `research_only=true` receiver corpus, with a same-archive `[macOS-CPU advisory]` measurement of `d_seg=0.60198647` and `d_pose=163.11865234`; it is not the description stream of the solved pointer object. Therefore this run does **not** measure or close solved-object temporal coding. Exact blocker: `SOURCE_S4_NOT_SOLVED_POINTER_OBJECT_DESCRIPTION_STREAM`.

Pointer delta: **none**. `0.1910828242 [contest-CPU]` is unchanged.

## Custody and the three byte surfaces

The source object is the settled S4 archive, 451,191 B, SHA-256 `d84f2fe053239d1542ba381420e9569d431ed2015e22e60e49ef48f1321696ed`. Its `0.bin` is 1,285,943 B, SHA-256 `595e69d41f96cc1a33ca7b58c0ed386549bfda6389a8176b24d7044d1f55955b`.

The often-quoted 216,207 B is not an archive and not all logical sections. It is only `base.pbase3` (36,011 B) plus `components.pcomp3` (180,196 B). The seed, events, and container bytes are already inside the 451,191 B archive and must not be added again.

The exact deterministic projection is:

| surface | before | after | delta |
|---|---:|---:|---:|
| description ledger: base + components | 216,207 B | 224,715 B | +8,508 B |
| full one-member deterministic ZIP | 451,191 B | 460,168 B | +8,977 B |
| description versus 216,300 B phase box | — | 224,715 B | +8,415 B |
| description versus 154,600 B phase box | — | 224,715 B | +70,115 B |

The projected archive is physically materialized at `/Volumes/VertigoDataTier/pact/evidence/xi_temporal_574_20260721/projected_full_archive/archive.zip`, 460,168 B, SHA-256 `5eac6976aaa8f56f807c9a4fa60e9b10015db381d06c7c5dace03fd86cd6c388`. Its S4 container parses back exactly and its XTDL1 payload decodes through the real repository decoder. It is **not standalone receiver-closed**. No standalone probe was run; unsupported XTDL1 dispatch is derived from the pinned S4 runtime source, SHA-256 `eef055896474b8327baf57ace016c37fe651f4c22534e2442ebc44da8c3f40b0`. This is a projection, not a candidate archive or score claim.

The invalid shortcut `451191 - 216207 + D_new` was refused because outer deflate couples all section bytes. The measured law is `A(theta)=len(DetZip9(SerializeS4(updated_manifest, seed, base_xtdl1, causal, events, components)))`.

## Corrected xi and coder

The source is the 600 BEV-v2 exact cross-pair and within-pair stages. For `t>0`:

`xi[t] = log_se3(exp_se3(xi_cross[t]) @ exp_se3(xi_within[t]))`.

Pair zero is the temporal keyframe. Translation scale 0.16, rotation scale 1.0, and pitch -0.05 are recalled from the sealed full-screw LawRefs. All six coordinates are nonzero on 599/600 rows. The fp64 xi stream SHA-256 is `5269b47a0fb5d8cb10dabc860da8be05511cebf396861e9a0aac35267a23462d`.

XTDL1 consumes the exact settled coherent-slot `(Q, presence, K, base_steps)` lattice directly, without decoding and re-sorting Lane lines. It predicts the next quantized Lane row from the previous decoded row plus `(rho_z, rho_x, omega_y)`, maps signed innovations through zigzag/uvarint, and routes every residual byte through `tac.shared_pmf_model` over the canonical range coder. Context bins `{1,2,4,8}` are all encoded and the smallest complete XTDL1 wire wins; n600 selected one bin.

The n600 planar-3 bundle is 44,399 B before terminal LZMA: 4,994 B counted six-coordinate xi payload, 318 B shared-PMF model, 36,732 B range payload, 450 B presence, and 1,905 B strict header/framing/digests. Its re-derived shared-PMF estimate is 36,708 payload bytes. Even the identity-predictor control grows by 7,020 terminal bytes, and planar-3 prediction adds another 1,488 terminal bytes; both counted context/full-xi cost and misprediction matter.

The wire fails closed on exact header keys/types, sizes, canonical JSON, segment hashes, outer digest, xi and shared-PMF decode/re-encode identity, re-derived entropy accounting, signed-int64 predictor arithmetic, varint consumption, exact slot-grid semantics, and malformed Brotli/zlib segments. The focused suite has 21 passing tests. Live disk free-space is still fail-closed and recorded in `checkpoints/storage_preflight.json`, but is excluded from the deterministic scientific receipt; two unchanged-source resumptions reproduced receipt SHA-256 `2c704c5df77e20bf0b7dbd83c9a492f05cfc8323a6e65bc880480d3524a33fd4` and archive SHA-256 `5eac6976aaa8f56f807c9a4fa60e9b10015db381d06c7c5dace03fd86cd6c388` byte-for-byte.

## Stream and per-stratum inventory

- PPCS seed 884,872 B: unchanged. Its counted planar trajectory is not the corrected full screw.
- PXQ1 static quotient 610 B: unchanged; null/range projection is already upstream.
- LBND2 Lane 35,393 B terminal / 159,386 B decoded: treatment target; coherent-slot front end already present.
- PCR3 causal: empty selected policy.
- PCE3 events 181,904 B / 942,475 B decoded: unchanged; already adjacent-frame LAP/XOR INTER coding.
- PCOMP3 components 180,196 B / 1,155,245 B decoded: unchanged; a semantic persistent-ID encoder is a separate owed parser surface.
- Movable: no admitted class-3 PCOMP3 packets, so `movable_site_coder.py` is recalled but not falsely credited.

This implementation does not recode PPCS seed, PCE3 events, PCOMP3 persistent objects, or Movable sites. Those non-Lane temporal streams remain unimplemented and unmeasured here.

Exact unchanged PCOMP3 unique-home bytes are Road/cell 130,736; Road/boundary 829; Lane/cell 7,803; Lane/boundary 359; Undrivable/cell 39,406; Undrivable/boundary 273; MyCar/cell 790. LBND is a shared analytic Lane generator and current custody does not uniquely split it into V9 cell/boundary/critical bytes.

The mandated `tools/measure_per_stratum_recursive_fractal_optimal.py` is absent at delegation base `74324e81fa`. The later tool's settled control consumes quarantined M1 bytes, so this lane did not recreate, cherry-pick, or silently substitute it. Exact blocker: `CANONICAL_PER_STRATUM_TOOL_ABSENT_AT_DELEGATION_BASE`; V9 remains `NO_VERDICT_RECEIVER_RATE_CUSTODY` until MAIN reviews and wires this receipt into the canonical surface.

## Routing

Do not promote or integrate XTDL1 into this S4 LBND chart: it loses at n64 and n600. Keep settled LBND2; the already-built LBND4 entropy-only arm is a distinct non-xi control. The solved-object description stream was not available in this source artifact, so it remains explicitly unmeasured. Reopen temporal description coding on that solved-object stream, on a chart that has not absorbed ego motion, or after landing public semantic encoders for PCE3/PCOMP3 with persistent component identity and exact receiver integration.

DSL leg: no new lever is admitted because the formulation loses and standalone receiver custody is absent. DAG leg: `.omx/research/xi_temporal_delta_coder_574_DAG_FEED_20260721T222234Z.md`. Equations leg: `.omx/research/xi_temporal_delta_coder_574_canonical_equations_20260721T222234Z.md`.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; v7.5/v8 specs; `reports/latest.md`; `.omx/state/lane_registry.json`; `.omx/state/subagent_progress.jsonl`; Task #574 per-arm and broadcast inboxes; settled S4 archive/receipt; BEV-v2 n64/n600 receipts and 600 pose stages; full-screw receipt/LawRefs; `analytic_lane_render_band.py`; `lane_track_and_smooth.py`; `shared_pmf_model.py`; `codec_pipeline.py`; `pr91_hpm1_range_contract.py`; `movable_site_coder.py`; recursive-fractal design and per-stratum signals.

MAIN landing review is mandatory before merge.
