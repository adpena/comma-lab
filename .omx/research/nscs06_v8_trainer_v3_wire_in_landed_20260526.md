# NSCS06 v8 trainer `cls_bytes` routing — wire-in LANDED

- **subagent_id**: `nscs06-v8-trainer-v3-wire-in-cls-bytes-routing-20260526`
- **lane_id**: `lane_nscs06_v8_trainer_v3_wire_in_cls_bytes_routing_20260526` (L1)
- **landed_utc**: 2026-05-26T18:50:00Z
- **scope**: code wire-in only; LOCAL macOS M5 MAX; **NO PAID DISPATCH** per operator standing "Remember all on MLX"
- **execution_cost**: $0 GPU + ~30 min wall-clock
- **horizon_class**: `plateau_adjacent` (Catalog #309)
- **predicted_band**: N/A (no empirical training); downstream paired-Modal verifies the empirical question per T3 #1335 WINNER #1
- **evidence_grade**: `predicted` (structural-byte-invariant tests only)
- **score_claim**: False; **promotable**: False; **axis_tag**: `[prediction]`

## Pre-execution gate verdict

PROCEED per `.omx/research/nscs06_v8_trainer_v3_wire_in_pre_execution_gate_report_20260526.md`. Scope bounded to 1 trainer file (Stage 5b insert + 1 kwarg add) + 1 new test file + 2 memos. Codec + inflate surfaces UNTOUCHED (already wired by sister commits `581b7b129` + `545beb35c`).

## Wire-in summary (file/line/diff)

### `experiments/train_substrate_nscs06_v8_chroma_lut.py`

**1. NEW Stage 5b** (lines 730-754; inserted after pose_quantized stage):

Derives per-cell SegNet class labels at low-res via NEAREST downsample of `cls_full` (Stage 4 output: full-resolution `(n_pairs, 384, 512)` SegNet argmax). NEAREST downsample = point-sample top-left pixel of each `grayscale_downsample`-sized cell; canonical sister of inflate.py's `Image.NEAREST` upsample. Shape invariant enforced: `cls_lowres.shape == (n_pairs, h_g, w_g)` matches `gray_lowres.shape`.

```python
cls_lowres = cls_full[
    :,
    : h_g * args.grayscale_downsample : args.grayscale_downsample,
    : w_g * args.grayscale_downsample : args.grayscale_downsample,
]
if cls_lowres.shape != (n_pairs, h_g, w_g):
    raise RuntimeError(...)
cls_bytes = np.ascontiguousarray(cls_lowres, dtype=np.uint8).tobytes()
cls_lowres_sha = _sha256_bytes(cls_bytes)
_stage(f"cls_lowres_nearest_downsample_shape_{cls_lowres.shape}_sha_{cls_lowres_sha[:8]}")
```

**2. v2 callsite at line 770 routed through `cls_bytes`** (was: pre-wire-in v2 emission; now: v3 emission):

```python
bin_bytes = pack_archive(
    num_pairs=n_pairs, grayscale_h=h_g, grayscale_w=w_g,
    output_height=CONTEST_RAW_HW[0], output_width=CONTEST_RAW_HW[1],
    pose_bytes=pose_bytes, grayscale_bytes=gray_lowres.tobytes(),
    pose_quant_scale=args.pose_quant_scale,
    chroma_seed=seed,
    cls_bytes=cls_bytes,                            # NEW
)
archive_variant_tag = "v3_procedural_seed_with_cls_stream"   # was "v2_procedural_seed"
```

**3. v1 callsite UNTOUCHED**: v1 inline-LUT branch is codec-incompatible with v3 cls_stream (archive.py line 353: `ValueError: cls_bytes supplied but schema_version resolved to v1/v2`). v1 keeps the cargo-cult #5 `cls=0 uniform` legacy inflate fallback.

### `src/tac/substrates/nscs06_v8_chroma_lut/tests/test_trainer_v3_wire_in.py` (NEW)

9 dedicated tests covering:

- Stage 5b NEAREST downsample shape + uniform-class invariant + round-trip with Pillow NEAREST upsample
- Trainer's v2_procedural_seed branch (post-wire-in) emits schema_version=3
- cls_lowres shape matches grayscale_lowres shape (per-cell field contract)
- cls_bytes round-trips byte-identically through pack/parse
- Regression guard: NO cls_bytes ≠ v3 (catches future silent wire-in revert)
- Rate-axis byte cost invariant: `len(v3) - len(v2) == 4 + len(cls_bytes)`
- Catalog #233 4-gate REFRESH: end-to-end trainer-emit → codec-parse → inflate-consume coherence

## Round-trip empirical test results

**All 206 nscs06_v8_chroma_lut substrate tests PASS** (197 prior + 9 new):

```
src/tac/substrates/nscs06_v8_chroma_lut/tests/test_cls_stream_wire_in.py  17 passed
src/tac/substrates/nscs06_v8_chroma_lut/tests/test_mlx_iteration.py       31 passed
src/tac/substrates/nscs06_v8_chroma_lut/tests/test_path_3_c_prime...      44 passed
src/tac/substrates/nscs06_v8_chroma_lut/tests/test_revisions.py           56 passed
src/tac/substrates/nscs06_v8_chroma_lut/tests/test_substrate.py           49 passed
src/tac/substrates/nscs06_v8_chroma_lut/tests/test_trainer_v3_wire_in.py   9 passed (NEW)
============================== 206 passed in 2.73s ==============================
```

Specifically: the sister test `test_inflate_v3_vs_v2_produces_different_frames_proves_cls_consumption` (already passing pre-wire-in) is the canonical inflate-side proof that cls_stream is operationally consumed; the new `test_catalog_233_4_gate_refresh_trainer_codec_inflate_coherent` test wires trainer emission → codec parse → inflate consumption end-to-end and verifies a non-empty raw output.

## Catalog #233 4-gate REFRESH evidence

| Gate | Status post-wire-in | Evidence |
|---|---|---|
| 1. Smoke green | REFRESH-COMPLETE | trainer emits v3 archive cleanly; 206/206 substrate tests PASS; inflate consumes v3 end-to-end |
| 2. Tier-C MDL density measured | UNCHANGED (Phase 2 BUILD landing 2026-05-21) | downstream `tools/mdl_scorer_conditional_ablation.py --tier c` invariant; trainer plumbing change does not regress |
| 3. 100ep auth-eval anchor | OPERATOR-ROUTABLE-NEXT | paired Modal T4 4-arm dispatch per Catalog #246 (NOW UNBLOCKED) |
| 4. Custody per Catalog #127 | UNCHANGED | trainer continues to route through canonical `gate_auth_eval_call` (Catalog #226) |

Per CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch": this wire-in is a CARGO-CULT-UNWIND per the NSCS06 v6→v7 pattern (the assumption "v2 procedural_seed is sufficient" was empirically falsified by FAIL_AT_CLASS_1 per Path 3 #C' L0 scaffold; the unwind is `cls_stream consumed at inflate`).

## `## Drift surface declaration` (per NEW MLX↔CUDA bidirectional standing directive 2026-05-26)

| Surface | Determinism | Drift risk |
|---|---|---|
| Stage 4 cls_full (SegNet argmax) | bit-identical under bit-identical weights+inputs | **None** for cls_full; argmax is deterministic. SegNet weights load via canonical helper. |
| Stage 5b NEAREST downsample | pure numpy slicing | **None** — byte-stable across CPU/CUDA/MLX |
| Stage 6 chroma_lut SHA → seed_bytes (32) | hashlib.sha256 deterministic | **None** |
| Stage 9 pack_archive cls_bytes serialization | uint8 byte stream | **None** — byte-stable |
| Inflate cls_full upsample (Image.NEAREST) | pure-CPU Pillow | **None** (Pillow consistent across substrates) |

Mostly N/A: this wire-in is uint8 byte-stream plumbing. The only drift surface is upstream — SegNet weight loading + argmax determinism — both covered by existing canonical helpers per CLAUDE.md "eval_roundtrip" + "Differentiable scorer preprocess" non-negotiables.

## `## Canonical-vs-frontier-push decision` (per NEW pushing-the-frontier-of-research-on-optimization-algorithms standing directive 2026-05-26)

**CANON-APPLICATION**. This is trainer plumbing routing existing canonical surfaces:

| Layer | Canonical-vs-frontier decision |
|---|---|
| Stage 4 cls_full computation | ADOPT_CANONICAL — already canonical SegNet argmax via `tac.scorer.load_differentiable_scorers` |
| Stage 5b NEAREST downsample | ADOPT_CANONICAL — numpy slicing is the canonical sister of Pillow Image.NEAREST upsample |
| Stage 9 pack_archive(cls_bytes=) | ADOPT_CANONICAL — `pack_archive` kwarg landed sister commit `581b7b129` |
| Inflate v3 branch | ADOPT_CANONICAL — `if arc.cls_lowres is not None:` branch landed sister commit `581b7b129` |

No new algorithm; no new optimization theory; no MLX iteration depth change. The frontier work is the EMPIRICAL paired-Modal-T4 4-arm question (now unblocked by this wire-in), NOT this code change.

## 6-hook wire-in declaration (per Catalog #125)

| Hook | Status | Rationale |
|---|---|---|
| 1. Sensitivity-map | N/A | trainer plumbing; no signal contribution |
| 2. Pareto constraint | N/A | no Pareto-relevant signal |
| 3. Bit-allocator | N/A | rate-axis cost is canonical equation #26 + cls_stream ADDITIVE (already documented in archive.py docstring) |
| 4. Cathedral autopilot dispatch | **ACTIVE** | unblocks T3 #1335 WINNER #1 NSCS06 v8 chroma_lut paired-Modal T4 4-arm dispatch per Catalog #246 |
| 5. Continual-learning posterior | **ACTIVE** | post-dispatch empirical anchor will register via `tac.cost_band_calibration.append_anchor` per canonical paired-dispatch flow |
| 6. Probe-disambiguator | N/A | the canonical disambiguator is the sister `test_inflate_v3_vs_v2_produces_different_frames_proves_cls_consumption` (already landed) |

## META-pattern surface — proposed STRICT preflight gate

**Bug class**: "codec API extended (new kwarg / new schema version) but trainer entry point not routed through it". Anchor: this exact session — sister cls_stream wire-in commits `581b7b129` + `545beb35c` extended `pack_archive(cls_bytes=)` + parse_archive v3 dispatch + inflate v3 branch + 17 tests, but `experiments/train_substrate_nscs06_v8_chroma_lut.py::_full_main` continued emitting v2 archives. The codec surface was structurally complete; the trainer surface silently lagged.

**Proposed Catalog #N** (numbered post-claim): `check_substrate_trainer_routes_through_latest_codec_schema_version`

Refuses substrate trainers under `experiments/train_substrate_*.py` that call `pack_archive(...)` whose target codec module declares a SCHEMA_VERSION constant strictly greater than the schema_version emitted by ANY trainer callsite, UNLESS:
- Same-line waiver `# SUBSTRATE_TRAINER_NOT_AT_LATEST_SCHEMA_OK:<rationale>` (≥4 chars; placeholder rejected per Catalog #287)
- Trainer declares `lane_class=substrate_engineering` per HNeRV parity L7
- Trainer is `research_only=true` per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"

Sister of Catalog #233 (L1→L2 promotion canonical 4-gate; covers the substrate-class promotion surface) + Catalog #240 (recipe-vs-trainer-state consistency; covers the recipe-trainer boundary) + Catalog #146 (Phase 1 trainer contest-compliant runtime emission; covers the trainer-runtime contract).

Together they extinct the "codec API extended but trainer not routed" bug class STRUCTURALLY at FOUR surfaces: codec contract (existing codec test patterns) + L1→L2 promotion gate (#233) + recipe-trainer consistency (#240) + trainer-codec schema alignment (#N proposed).

**Operator-routable**: claim Catalog #N via `tools/claim_catalog_number.py claim --commit-via-serializer --reason "trainer_codec_schema_alignment_gate"` in a follow-up subagent if this META-pattern recurs (one anchor is empirical evidence; recurrence + sister anchors would justify the new STRICT gate per Catalog #299 "stop and consolidate" pause discipline; current count 361 well under 400 quota).

## Operator-routable next steps

1. **Paired Modal T4 4-arm dispatch NOW UNBLOCKED** per Catalog #246. The T3 #1335 WINNER #1 NSCS06 v8 chroma_lut ordering can now fire its empirical anchor:
   - Recipe: `.omx/operator_authorize_recipes/substrate_nscs06_v8_chroma_lut_modal_*.yaml`
   - Smoke-before-full per Catalog #167
   - Cost band: ~$0.50-1.00 paired auth_eval per T3 #1335 prediction
   - Verifies empirical question: does v3 cls_stream consumption (cargo-cult #5 unwind) produce the predicted seg+pose-axis improvement that offsets the cls_stream rate-axis cost (~+0.0049 at realistic shapes)?
2. **Update lane registry** per Catalog #233 4-gate REFRESH (trainer + codec + inflate coherent; gates 2-4 unchanged from Phase 2 BUILD baseline; this landing refreshes Gate 1 smoke green).
3. **NSCS06 v8 substrate trainer is now at OPTIMAL FORM** per CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" non-negotiable: cargo-cult #5 unwound at codec + inflate + trainer surfaces simultaneously.

## Sister coordination (Catalog #230 + #340)

- **Slot 2** (Z7-Mamba-2 L1 EMPIRICAL fair-shake) — different substrate scope; no file overlap; no checkpoint collision
- **Slot 3** (BoostNeRV BPR1 Variant B codec redesign) — different substrate scope; no file overlap; no checkpoint collision
- **My scope**: `experiments/train_substrate_nscs06_v8_chroma_lut.py` (1 file) + `src/tac/substrates/nscs06_v8_chroma_lut/tests/test_trainer_v3_wire_in.py` (NEW) + 2 research memos

Catalog #340 sister-checkpoint guard: clean (no in-flight sister on my files within 60-min window).

## Discipline

- Catalog #229 PV: read trainer Stage 4/5/5b/9 in full + archive.py + inflate.py + test_cls_stream_wire_in.py before edit
- Catalog #287 placeholder-rationale rejection: all waivers in new test/code use substantive ≥4-char rationales
- Catalog #340 sister-checkpoint guard: PROCEED verified
- Catalog #343 no hardcoded frontier-band score literals: this memo cites NO `0.19xxx` / `0.20xxx` literals
- Catalog #206 checkpoints: 2 in-progress + 1 complete-at-landing
- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE: NEW landing memo + NEW test file + NEW research memo; sister cls_stream wire-in landing memos NEVER mutated
- Canonical serializer + POST-EDIT `--expected-content-sha256` per Catalog #117/#157/#174
- Per CLAUDE.md "Forbidden premature KILL": no KILL verdicts; trainer wire-in is iteration toward OPTIMAL FORM per CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" non-negotiable

## Mission alignment per Catalog #300

**Predicted mission contribution**: `frontier_protecting` + `frontier_breaking_enabler` (composite). Frontier-protecting because this prevents the future-recurrence of the codec-vs-trainer-schema-drift bug class (canonical alignment now structural; sister META-pattern STRICT gate proposed). Frontier-breaking-enabler because this UNBLOCKS T3 #1335 WINNER #1 NSCS06 v8 chroma_lut paired-Modal T4 4-arm dispatch which carries the predicted ΔS band per T3 council's empirical verdict.

[predicted; canonical-equation-N/A; structural-byte-invariant-only; per-substrate-symposium-pending paired Modal T4 dispatch]
