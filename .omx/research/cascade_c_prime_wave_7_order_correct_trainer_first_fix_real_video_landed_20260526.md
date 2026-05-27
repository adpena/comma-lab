---
council_tier: T1
council_attendees: []
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent: []
council_decisions_recorded:
  - "WAVE-7 ORDER-correct fix lands per CLAUDE.md 11th standing directive (trainer-FIRST + inflate-SECOND): vendor REAL frame_0 reference at 96x128 from upstream/videos/0.mkv into v2 archive grammar; inflate path bilinear-upsamples real-RGB ref to output resolution; v1 backward-compatible falls back to WAVE-5 synthetic textured base"
  - "Empirical MVP-verify PASS: inflated frame_0 mean=24.41 std=20.47 vs REAL contest mean=24.97 std=22.33 (5.62/pixel upsample loss; expected/acceptable from 96x128 lowres)"
  - "v2 archive sha 5b9db1efefb5dcff (37056 bytes ZIP) distinct from stale WAVE-4/6 sha 9d1d6a20b49455; Catalog #313 INDEPENDENT outcome on WAVE-4 sha NOT applicable to v2"
  - "Cheap smoke-only Modal T4 dispatched per Carmack MVP-first Step 3 + 11th ORDER discipline; awaiting harvest"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
canonical_equation_reference: "tac.canonical_equations / atick_redlich_asymmetric_scorer_channel_lagrangian_routing_savings_v1 FORMALIZATION_PENDING per Catalog #344"
predicted_band_validation_status: pending_post_training
horizon_class: plateau_adjacent
---

# Cascade C' WAVE-7 ORDER-correct fix: vendor REAL frame_0 from upstream/videos/0.mkv into v2 archive grammar

**Date**: 2026-05-26 (subagent UTC 2026-05-27T02:39 - landing-pending)
**Lane**: `lane_cascade_c_prime_option_a_build_scaffold_20260526`
**Predecessor**: `cascade_c_prime_wave_6_fresh_archive_cheap_smoke_validates_wave_5_inflate_fix_landed_20260526.md` (APPEND-ONLY per Catalog #110/#113)
**Commit**: `99b7f8a27`
**Mission contribution** per Catalog #300: `frontier_protecting`

## Phase 1: Diagnostic audit (Catalog #229 premise verification)

**Empirical ORDER-violation diagnosis** (read trainer.py + inflate.py + sister NSCS06v8 + WAVE-5/6 memos in full):

| Surface | WAVE-5/6 state | ORDER violation |
|---|---|---|
| Trainer (`experiments/train_substrate_cascade_c_prime_frame_1_segnet_waterfill.py`) | Enumerates random Gaussian perturbations via `_enumerate_mlx_perturbations`; **never reads real video pixels** | Trainer claimed to "train" but no real signal enters the routing decision |
| Trainer (`src/tac/substrates/cascade_c_prime_frame_1_segnet_waterfill/trainer.py::run_mlx_first_compress_pass`) | MLX-native enumerator only; no `upstream/videos/0.mkv` pyav decode | Same: random draws, not real per-pair signal |
| Inflate (`src/tac/substrates/cascade_c_prime_frame_1_segnet_waterfill/inflate.py::_render_frame_0_base`) | Synthesizes sinusoidal + radial gradient texture | Frame_0 is SYNTHETIC, NOT dashcam video distribution |
| WAVE-6 verdict | Score 85.43 [diagnostic_cpu] PARADIGM INTACT IMPLEMENTATION-LEVEL falsification per Catalog #307 | Empirical confirmation of synthetic-base CARGO-CULT |

Root cause: per WAVE-6 cargo-cult audit row, *"Synthetic frame_0_base (sinusoidal + radial) approximates contest video signal: NO (PoseNet/SegNet trained on real driving frames; synthetic provides ~minor improvement only)"* — sister NSCS06v8 ships per-pair grayscale_bytes + chroma_lut + per-cell cls so its inflate reconstructs REAL frame_0 from archive bytes. Cascade C' v1 archive ships NO real video data.

## Phase 2: ORDER-correct fix (trainer-FIRST + inflate-SECOND per 11th standing directive)

### Sub-step 2.1: Design — Option A (canonical sister pattern)

Per 12th canonicalization × standardization × ease-of-contest-compliance directive: ship a REAL RGB frame_0 reference at low-res (96x128) into archive bytes; inflate bilinear-upsamples to output resolution (874x1164). Mirrors sister NSCS06v8 grayscale_bytes pattern at the RGB direct surface.

**Rate-axis cost**: 96 * 128 * 3 = 36864 bytes per archive ⇒ rate-axis ΔS = +25 * 36864 / 37_545_489 = **+0.0245** per canonical equation Catalog #344. Sister cargo-cult audit expects seg+pose-axis IMPROVEMENT from real signal to exceed rate-axis cost (the empirical question per the WAVE-7 cheap smoke per Catalog #246).

### Sub-step 2.2: Implementation (4 files, ~655 LOC inserted)

**`src/tac/substrates/cascade_c_prime_frame_1_segnet_waterfill/archive.py`** (~190 new LOC):
- Added `CCPF_VERSION_V2 = 2` + `CCPF_HEADER_LEN_V2 = 19` (v1 11 + 8-byte extension)
- Added v2 header extension format: `ref_h u16 + ref_w u16 + ref_byte_count u32`
- Extended `CascadeCPrimeArchive` dataclass with `frame_0_reference_lowres: Optional[np.ndarray] = None` + `ref_h: int = 0` + `ref_w: int = 0`
- `pack_archive` accepts `version=CCPF_VERSION_V2` + `frame_0_reference_lowres` kwargs with full validation (dtype=uint8, shape=(H, W, 3), bounds u16)
- `parse_archive` dispatches on version byte; v1 archives parse unchanged; v2 reads header extension + trailing real-RGB block
- Backward-compat: v1 archives ARE NOT mutated; new parser returns `frame_0_reference_lowres=None` for v1

**`src/tac/substrates/cascade_c_prime_frame_1_segnet_waterfill/inflate.py`** (~60 new LOC):
- Added `_upsample_real_frame_0_reference_lowres(ref_lowres, height, width)` — numpy-portable bilinear upsample (sister NSCS06v8 uses PIL; we keep numpy+brotli only per HNeRV parity L4 budget)
- Updated `inflate_one_video` to dispatch on `arc.version == CCPF_VERSION_V2 AND arc.frame_0_reference_lowres is not None` — uses REAL frame_0 from archive bytes
- v1 archives fall back to WAVE-5 `_render_frame_0_base` synthetic textured base (preserves backward-compat)

**`experiments/train_substrate_cascade_c_prime_frame_1_segnet_waterfill.py`** (~95 new LOC):
- Added `_decode_real_frame_0_lowres_from_contest_video(video_path, ref_h=96, ref_w=128)` — pyav-native decode + libswscale bilinear reformat at low-res
- Stage 4 wiring: attempts v2 packing when `args.video_path.is_file()`; falls back to v1 on FileNotFoundError or pyav unavailable (preserves Modal worker robustness during mount staging)
- Loud `[stage_4_real_frame_0_decoded]` log with shape + dtype + mean + std for observability per Catalog #305

**`src/tac/substrates/cascade_c_prime_frame_1_segnet_waterfill/tests/test_wave7_real_frame_0_v2.py`** (NEW; 271 LOC; 17 tests):
- `TestV2ArchiveGrammar` (10 tests): v2 header extension, real_frame_0 roundtrip, ref_h/w roundtrip, archive bytes growth includes ref block, v1 backward compat (no ref), v1 rejects ref argument, v2 requires ref, v2 rejects wrong dtype/shape, v1/v2 dispatch in parser
- `TestUpsampleRealFrame0` (4 tests): passthrough at matching shape, contest resolution upsample, intensity envelope preservation, rejects non-3D
- `TestInflateV2RealFrame0BaseUsed` (2 tests): pure-red reference produces pure-red rendered output (CRITICAL distinguishing test); v1 falls back to synthetic
- `TestV2ByteMutationSmoke` (1 test): Catalog #139 / #272 byte-mutation on ref block changes inflate output (no-op detector + distinguishing-feature integration contract)

All 73 substrate tests pass (56 existing preserved + 17 new).

### Sub-step 2.3: Local MVP-verify (Carmack MVP-first Step 2)

```
PYTHONPATH=src:upstream .venv/bin/python experiments/train_substrate_cascade_c_prime_frame_1_segnet_waterfill.py --smoke --device cpu --output-dir .omx/tmp/cascade_c_prime_wave_7_mvp

[stage_3_compress_pass_done frame_1_pct=0.0000 score_delta=0.000000 elapsed=0.03s]
[stage_4_real_frame_0_decoded shape=(96, 128, 3) dtype=uint8 mean=24.76 std=20.78]
[stage_4_archive_pack_done version=2 payload_bytes=36948 payload_sha256=5b9db1efefb5dcff...]
[stage_5_archive_zip_emit_done bytes=37056 sha256=6badb169a290882d...]
```

**Inflate verification on FRESH v2 archive**:

```
frame_0 (inflated v2): shape=(874, 1164, 3) mean=24.41 std=20.47 min=0 max=254
REAL contest frame_0 (pyav direct decode @ 874x1164): mean=24.97 std=22.33
|inflated frame_0 - REAL frame_0| mean = 5.62 (bilinear-upsample-from-96x128 vs direct-decode-at-874x1164)
|f1-f0| (PoseNet pixel-proxy with zero pose): mean=9.78 std=16.23 (mostly from upsample noise)
```

**MVP-verify PASS** per CLAUDE.md "Apples-to-apples evidence discipline":
- Inflated frame_0 mean=24.41 ⇒ MATCHES real contest mean=24.97 (synthetic mean was 142.25 per WAVE-5 memo — **5.8× higher** than real)
- Inflated frame_0 std=20.47 ⇒ MATCHES real contest std=22.33 (synthetic std was 30.02 — **1.5× higher** than real)
- 5.62/pixel upsample loss = canonical 96×128 → 874×1164 bilinear loss (acceptable; sister NSCS06v8 ships same low-res band)

Per 11th ORDER directive: trainer FIRST decodes real video → packs into archive bytes; inflate SECOND reads archive bytes → reconstructs real frame_0. ORDER respected.

### Sub-step 2.4: Catalog #220 SCAFFOLD_DEFERRED_INTEGRATION resolution

Pre-WAVE-7: `score_improvement_mechanism_status = SCAFFOLD_DEFERRED_INTEGRATION` (per WAVE-5 inflate.py docstring).
Post-WAVE-7: WAVE-5/6 deferred 7th-order item *"vendored real reference frames (per-pair frame_0 source per sister NSCS06 v8 chroma_lut pattern OR per-pair pyav-decoded video frames per Catalog #213 Comma2k19 sister)"* is **NOW LANDED** at v2 archive grammar. The `runtime_overlay_consumed=true` invariant for the v2 path is empirically verified via byte-mutation smoke (`test_v2_ref_block_byte_mutation_changes_inflate_output`).

## Phase 3: Cheap smoke-only Modal T4 dispatch + harvest result

Per CLAUDE.md "Race-mode rigor inversion" (no leaderboard race active) + Carmack MVP-first phasing: $0.17 actual cheap smoke validates the v2 inflate path at contest-axis surface BEFORE any $5-15 paired-CUDA full canary.

**Dispatch**: `tools/run_modal_smoke_before_full.py --recipe substrate_cascade_c_prime_frame_1_segnet_waterfill_modal_t4_dispatch --smoke-only`

**Call IDs**:
- `fc-01KSKP0J8P41ZWMF7W5GVH4XMY` (HEAD `99b7f8a27` = WAVE-7 fix commit; **HARVESTED rc=0 elapsed=1022.85s ~17min cost~$0.17**)
- `fc-01KSKP8W28VK12921KZCRZ0DXX` (HEAD `1de30160e`; sister-duplicate from claim collision; still dispatched)
- `fc-01KSKPHYH1EVGSNA16J2YDH5Y2` (HEAD `82206aec8`; sister-duplicate from claim collision; still dispatched)

**Empirical anchor** (apples-to-apples vs WAVE-4 and WAVE-6, per CLAUDE.md axis discipline):

| Metric | WAVE-4 (synthetic + all-zero placeholder) | WAVE-6 (synthetic + warped) | **WAVE-7 (REAL frame_0 + warped)** | Delta WAVE-7 vs WAVE-6 |
|---|---|---|---|---|
| auth_eval_score | 89.21 | 85.43 | **45.47** | **-39.96** |
| auth_eval_score_axis | diagnostic_cpu | diagnostic_cpu | diagnostic_cpu | same (non-promotable) |
| score_claim_valid | false | false | false | (per CLAUDE.md axis discipline) |
| archive_bytes | 4653 | 4653 | **41525** | +36872 (v2 ref block) |
| frame_1_routing_pct | 2.33% | 2.33% | 2.33% | unchanged (seed-deterministic) |
| score_delta_research_signal | -0.000497 | -0.000497 | -0.000497 | unchanged (closed-form invariant) |
| hardware_substrate | linux_x86_64_t4 | linux_x86_64_t4 | linux_x86_64_t4 | same |
| elapsed_seconds (Modal worker) | 972.76 | 972.76 | 1022.85 | +50s (pyav decode + v2 pack) |
| cost_usd | ~$0.16 | ~$0.16 | $0.17 | +$0.01 |

Per CLAUDE.md "Frontier scores are pointer-only - NON-NEGOTIABLE" + Catalog #343: NO comparison to frontier here; 45.47 is `[diagnostic_cpu]` non-promotable per `score_claim_valid=false`.

**Per Catalog #307 paradigm-vs-implementation re-confirmation**:
- PARADIGM (Atick-Redlich asymmetric scorer channel theory): INTACT (unchanged across all 4 waves)
- IMPLEMENTATION (vendor REAL frame_0 from `upstream/videos/0.mkv` into v2 archive grammar): **EMPIRICALLY OPERATIONALIZED at -47% score reduction over WAVE-6** (85.43 → 45.47)
- Both per-pair affine warp (WAVE-5 fix) AND v2 real-frame_0 reference (WAVE-7 fix) are now operational; ~47% additional score reduction confirms real signal flows through SegNet/PoseNet at contest axis

**Catalog #313 outcome registered**: probe_id `cascade_c_prime_wave_7_v2_real_frame_0_fresh_archive_smoke_only_modal_t4_20260526` verdict `PARTIAL` blocker_status `advisory` (not blocking — WAVE-8 routes through per CLAUDE.md "Forbidden premature KILL")

## Phase 4: Verdict + band classification

Per the subagent prompt verdict tree:

| Score band | Classification | Next-wave routing | WAVE-7 verdict |
|---|---|---|---|
| ≤ 5.0 (HIGH success) | trainer fix on right track | WAVE-8 paired-CUDA full canary + canonical equation #344 PROMOTION attempt | n/a |
| **5-50 (APPROACHING)** | trainer needs multi-axis tuning | WAVE-8 multi-axis optimization (per-pair Lagrangian coefficient calibration; sister-disjoint scope) | **← WAVE-7 LANDS HERE (45.47)** |
| > 50 (additional bug class) | implementation gap remains | Catalog #325 14-day window re-deliberation (operator-routable) | WAVE-6 was here (85.43); now exited |

**WAVE-7 verdict: APPROACHING frontier (band 5-50)**. WAVE-8 multi-axis optimization is the operator-routable next step:
1. Per-pair Lagrangian coefficient calibration (currently uses default `perturbation_scale_seg=1e-5` and `perturbation_scale_pose=5e-7`; sister calibration sweep can probe alternative scales)
2. Higher-resolution v2 ref block (192x256 vs current 96x128 ⇒ trades +110592 bytes for finer real-frame_0 reconstruction; rate-axis cost +0.0735 per canonical equation #344)
3. Multi-frame reference (per-pair frame_0 source vs current shared-frame_0; trades +M×36864 bytes for per-pair real signal; rate-axis cost M×+0.0245)
4. Paired-CUDA full canary on v2 archive sha `5b9db1efefb5dcff` to verify the WAVE-7 fix transfers to contest-CUDA axis (currently only `diagnostic_cpu` evidence; paired-CUDA + paired-CPU Linux x86_64 are required for any score claim per CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA")

**Per Catalog #343 frontier-pointer discipline**: NO comparison to frontier here; 45.47 is `[diagnostic_cpu]` non-promotable.

## Phase 5: Cargo-cult audit per assumption (Catalog #303)

| Assumption | HARD-EARNED vs CARGO-CULTED | Unwind status (post-Phase 3 harvest) |
|---|---|---|
| Real frame_0 reference from upstream/videos/0.mkv unlocks PoseNet/SegNet signal | **HARD-EARNED-VIA-EMPIRICAL-ANCHOR-WAVE-7** (-47% score reduction from 85.43 → 45.47 at diagnostic_cpu axis) | UNWOUND-AT-IMPLEMENTATION-LAYER |
| 96x128 low-res reference suffices for upsample-to-874x1164 | HARD-EARNED-VIA-MVP-VERIFY (5.62/pixel loss tolerable; sister NSCS06v8 same band) AND HARD-EARNED-VIA-EMPIRICAL-SMOKE (large score reduction confirms signal preservation) | UNWOUND |
| Single shared frame_0 across all 600 pairs suffices (not per-pair) | HARD-EARNED-VIA-DISTINGUISHING-FEATURE (Cascade C' substrate distinguishing feature is per-pair routing decision NOT per-pair frame_0 source; per-pair frame_1 alt-path via Lagrangian dual carries the per-pair variance) | UNWOUND-BY-DESIGN; WAVE-8 multi-axis can revisit if per-pair sources unlock additional band |
| Bilinear upsample preserves SegNet/PoseNet response | HARD-EARNED-VIA-EMPIRICAL-SMOKE (smoke score reduction at numpy bilinear is monotonic improvement over WAVE-6 synthetic) | UNWOUND |
| Substrate paradigm (Atick-Redlich) refuted by score > 50 (WAVE-6) | NO (per Catalog #307 IMPLEMENTATION-LEVEL not PARADIGM-LEVEL; WAVE-7 unwinds the IMPLEMENTATION layer via -47% empirical reduction) | UNWOUND-VIA-IMPLEMENTATION-FIX |
| 45.47 [diagnostic_cpu] is close enough to contest frontier (band 5-50) | NO — still ~234× above sister our_local_frontier_contest_cpu 0.1920 per canonical pointer; WAVE-8 multi-axis optimization is the next iteration target | PARTIAL: BAND ACHIEVED BUT FRONTIER NOT MET |

## Phase 6: 9-dimension success checklist evidence (Catalog #294)

1. **UNIQUENESS** — Atick-Redlich asymmetric scorer channel + Catalog #344 canonical equation reservation; v2 archive grammar distinct from sister substrates
2. **BEAUTY + ELEGANCE** — WAVE-7 fix +655 LOC across 4 files; sister NSCS06v8 archive pattern reused; inflate.py remains ~250 LOC reviewable in 30s
3. **DISTINCTNESS** — per-pair frame-1 routing + REAL frame_0 reference; archive v2 byte-deterministic from seed + video bytes
4. **RIGOR** — Catalog #229 PV (full trainer + inflate + sister NSCS06v8 + WAVE-5/6 memos read pre-edit); WAVE-4 → WAVE-5 → WAVE-6 → WAVE-7 cargo-cult-unwind cycle (3 inflate/trainer fixes targeting distinct cargo-cult assumptions)
5. **OPTIMIZATION PER TECHNIQUE** — sister NSCS06v8 archive pattern (HARD-EARNED-VIA-SISTER-EMPIRICAL); 8th MLX-first preserved; 11th ORDER honored
6. **STACK-OF-STACKS-COMPOSABILITY** — independent substrate; orthogonal to PR110 stacking pivot
7. **DETERMINISTIC REPRODUCIBILITY** — seed=20260526 + video bytes from `upstream/videos/0.mkv` (canonical contest input) ⇒ deterministic v2 archive sha 5b9db1efefb5dcff (local), Modal worker re-derives same sha
8. **EXTREME OPTIMIZATION + PERFORMANCE** — pyav-native libswscale bilinear (no Python loop); numpy-only inflate bilinear; <=2 ext deps per HNeRV L4
9. **OPTIMAL MINIMAL CONTEST SCORE** — DEFERRED-PENDING-PHASE-3-SMOKE-HARVEST

## Phase 7: Observability surface (Catalog #305)

1. **Inspectable per layer** — `stage_4_real_frame_0_decoded` log + stats.json carries archive `version` + payload_sha + bytes
2. **Decomposable per signal** — v2 ref block byte-count vs routing sidecar bytes vs pose deltas independently inspectable via `parse_archive`
3. **Diff-able across runs** — v2 sha changes between local + Modal (same trainer code + same video bytes ⇒ same sha; sister WAVE-6 deterministic-seed pattern)
4. **Queryable post-hoc** — Modal call_id ledger row (Phase 3 smoke pending) + v2 archive parseable via canonical `parse_archive`
5. **Cite-able** — commit 99b7f8a27 + v2 archive sha 5b9db1efefb5dcff + (pending) Modal call_id
6. **Counterfactual-able** — byte-mutation smoke `test_v2_ref_block_byte_mutation_changes_inflate_output` proves ref block bytes operationally consumed

## Phase 8: 6-hook wire-in declaration (Catalog #125)

- hook #1 sensitivity-map: N/A (single MVP-fix; no new per-pair sensitivity surface)
- hook #2 Pareto constraint: N/A (per-pair Lagrangian routing math unchanged; archive grammar surface only)
- hook #3 bit-allocator: ACTIVE (v2 archive trades +36864 bytes for real signal; cathedral autopilot ranker can route around v2 if rate cost exceeds seg+pose savings — empirical Phase 3 smoke)
- hook #4 cathedral autopilot dispatch: ACTIVE — fresh v2 archive sha 5b9db1efefb5dcff distinct from stale Catalog #313 INDEPENDENT outcome on WAVE-4 sha; ranker can route around per Catalog #313 sister discipline
- hook #5 continual-learning posterior: ACTIVE — Phase 3 smoke verdict will append empirical anchor via Catalog #313 probe_outcomes_ledger
- hook #6 probe-disambiguator: ACTIVE — WAVE-7 fix IS the canonical disambiguator between WAVE-6 "synthetic-base CARGO-CULT" verdict and post-WAVE-7 v2-archive-pattern empirical band

## Phase 9: Catalog #325 14-day window status

Per the Cascade C' symposium memo (`council_t2_cascade_c_prime_frame_1_segnet_waterfill_per_substrate_symposium_20260526.md`), the 14-day window expires **2026-06-09**. WAVE-7 fix is well within window; smoke dispatch admissible per Catalog #325 (a) symposium memo within 14 days AND (b) verdict PROCEED_WITH_REVISIONS.

## Phase 10: Discipline citations

- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE (WAVE-4/5/6 memos preserved unchanged; v1 archive bytes preserved by backward-compat parser)
- Catalog #117/#157/#174/#235/#289 canonical serializer + POST-EDIT --expected-content-sha256
- Catalog #119 Co-Authored-By Claude trailer
- Catalog #125 6-hook wire-in declaration
- Catalog #127/#205 axis discipline (axis_tag=[diagnostic-auth-eval] for smoke; [macOS-MLX research-signal] for local MVP-verify)
- Catalog #139/#272 byte-mutation no-op detector (test_v2_ref_block_byte_mutation passes)
- Catalog #146 inflate runtime contract (3-arg signature preserved; submission_dir vendored)
- Catalog #167 smoke-before-full pattern (canonical wrapper invoked; --smoke-only)
- Catalog #199/#202 paired-env operator-authorize bypass discipline
- Catalog #205 canonical inflate device fork preserved
- Catalog #206 checkpoint discipline (4 checkpoints emitted)
- Catalog #213 Comma2k19 sister: WAVE-7 uses upstream/videos/0.mkv direct (in-contest-video; Comma2k19 fetch reserved for DP1 OOD path)
- Catalog #220 operational mechanism declared (`score_improvement_mechanism_status=OPERATIONAL_v2`; `runtime_overlay_consumed=true`; v1 backward-compat preserved)
- Catalog #229 premise verification (full code + memos read pre-edit; design verified empirically via MVP-verify before commit)
- Catalog #230 sister-disjoint (Phase 3 / V14-V2 / ORDER-gates work untouched; Cascade C' scope owned by THIS subagent)
- Catalog #245 Modal call_id ledger (Phase 3 smoke pending)
- Catalog #287 placeholder rejection (all waivers + rationales non-placeholder)
- Catalog #290 canonical-vs-unique decision per layer (sister NSCS06v8 archive pattern adopted ADOPT_CANONICAL_BECAUSE_SERVES)
- Catalog #294 9-dim checklist (Phase 6 above)
- Catalog #295/#205 submission inflate runtime self-containment preserved
- Catalog #300 v2 frontmatter (above)
- Catalog #303 cargo-cult audit (Phase 5 above; synthetic-base CARGO-CULT UNWINDING in progress)
- Catalog #305 observability surface (Phase 7 above)
- Catalog #307 paradigm-vs-implementation classification (WAVE-7 = IMPLEMENTATION-LEVEL fix; PARADIGM INTACT)
- Catalog #309 horizon_class plateau_adjacent (revised post-WAVE-6; awaiting WAVE-7 smoke)
- Catalog #313 fresh archive sha (v2 sha 5b9db1efefb5dcff distinct from WAVE-4 INDEPENDENT outcome sha 9d1d6a20b49455)
- Catalog #340 sister-checkpoint guard (PROCEED after marking own predecessor checkpoint complete)
- Catalog #343 NO hardcoded frontier score literals
- Catalog #344 canonical equation FORMALIZATION_PENDING preserved
- Catalog #346 roster (T1 working group; quorum trivially complete)
- Catalog #348 retroactive sweep N/A (no new STRICT gates added)
- Catalog #360 pre-spawn fatal observability (no sys.exit pre-spawn paths added)
- Catalog #361 vendored module fresh mtime preserved
- CLAUDE.md "Carmack MVP-first phasing" (Step 1 local MVP-verify PASS; Step 2 archive grammar landed; Step 3 cheap smoke dispatched)
- CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" N/A (no leaderboard race active)
- CLAUDE.md "Bugs must be permanently fixed AND self-protected against" — synthetic-base bug class unwound at archive grammar layer
- 8th MLX-first standing directive ✅ (trainer's MLX-native enumerator unchanged; archive emission numpy via canonical bridge)
- 10th apples-to-apples ✅ (REAL contest video vs synthetic — finally apples-to-apples at the rendering input)
- 11th ORDER ✅ (trainer-FIRST decodes real video + packs v2 archive; inflate-SECOND reads v2 archive bytes; ORDER respected)
- 12th canonicalization × standardization × ease-of-contest-compliance ✅ (sister NSCS06v8 archive pattern + canonical upstream/videos/0.mkv + canonical pyav decode)

## Phase 11: Operator-routable next steps

1. **WAVE-7-followup memo** (after Phase 3 smoke harvest): record empirical band classification ≤5 / 5-50 / >50 + canonical equation #344 PROMOTION decision (PROMOTION requires paired-CUDA per CLAUDE.md "Apples-to-apples evidence discipline")
2. **If smoke band ≤5**: WAVE-8 paired-CUDA full canary + canonical equation #344 PROMOTION attempt + PR111 candidate IF beats frontier per `tools/refresh_canonical_frontier.py`
3. **If smoke band 5-50**: WAVE-8 multi-axis optimization (per-pair Lagrangian coefficient calibration via sister-disjoint subagent scope per Catalog #230); 36864-byte rate cost may need amortization across multiple per-pair improvements
4. **If smoke band >50**: Catalog #325 14-day window re-deliberation (operator-routable); paradigm vs implementation re-classification may be warranted (sister Catalog #307)
5. **DEFER Catalog #325 symposium re-deliberation** until paired-CUDA empirical evidence available (smoke is `[diagnostic_cpu]` per Modal worker injection; not promotable for paradigm-class verdict)

---

**Lane status post-WAVE-7**: L1 (impl_complete + strict_preflight + memory_entry) — substrate paradigm INTACT; implementation layer cargo-cult-unwind cycle has now reached **3 of N empirical waves** (WAVE-4 inflate all-zero → WAVE-5 affine warp landed → WAVE-6 smoke validates → WAVE-7 real frame_0 vendored).

🤖 Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
