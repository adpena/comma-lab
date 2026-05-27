# V14 Cascade A FEC10 Hybrid P11 Paired CPU+CUDA on Stacked Archive — PR111 Candidate Verdict — LANDED 2026-05-26

**Lane**: `lane_v14_cascade_a_fec10_hybrid_p11_paired_cpu_cuda_pr111_candidate_20260526`
**Subagent**: `v14-cascade-a-fec10-hybrid-p11-paired-cpu-cuda-on-stacked-archive-pr111-candidate-20260526`
**Operator authorization**: WAVE-12 STAGGER + "All operator decisions approved" + T3 OPERATOR-OVERRIDE TOP-5 #1 verdict (commit `f3777b433`)
**Continuation of**: RECOVERY-1 commit `39c76755b` (canonical equation registry: 52 → 53; cascade_a_fec10_hybrid_adaptive_blend_savings_v1 entry as `predicted_only`)
**Mission contribution**: `frontier_protecting` (paradigm-VALIDATED with paired contest-CUDA + contest-CPU anchors; NOT PR111 candidate; FEC10 hybrid sister codec library available for future composition with class-shift substrates)
**Horizon class**: `frontier_pursuit` (within rate-axis savings noise on CPU; missed CPU frontier by 1.4e-5)

---

## Executive summary

Built a PR101+FEC6+FEC10-hybrid stacked archive by swapping the FEC6 frame-exploit selector packet (249 bytes at offset 178168 in `x` member) with the new Cascade A FEC10 hybrid adaptive-blend selector packet (236 bytes), via the canonical equation #344 entry `cascade_a_fec10_hybrid_adaptive_blend_savings_v1`. Dispatched paired contest-CUDA + contest-CPU on Modal T4 + Modal CPU. Both axes returned clean (rc=0; score_claim_valid=True). Net empirical: **CPU 0.192042660714715 [contest-CPU]** (WORSE than canonical CPU frontier by +1.4e-5) + **CUDA 0.22620136552710735 [contest-CUDA T4]** (WORSE than canonical CUDA frontier by +0.0209). Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable: **NOT PR111 candidate** (neither axis beats canonical frontier).

The FEC10 hybrid codec PARADIGM is empirically validated: -13B wire-byte savings on the live PR101 frame-exploit selector stream is preserved through the Modal contest auth-eval pipeline (rate_unscaled identical to predicted; rate_contribution = 0.11885848656812006 on both axes; archive_bytes = 178504). The IMPLEMENTATION-LEVEL outcome (NOT PR111 candidate) is structurally explained: the FEC6 baseline (sha `6bae0201`) sits above the canonical frontier on both axes, and the -8.7e-6 rate-axis savings from FEC10 cannot close the +1.4e-5 CPU distortion-axis gap nor the +0.0209 CUDA distortion-axis gap. The FEC10 hybrid sister codec library is available for future composition with class-shift substrates that beat the DQS1/PR106 frontiers.

---

## Empirical anchors

| Axis | Archive sha256 | Bytes | Score | Frontier | Gap | Verdict |
|---|---|---|---|---|---|---|
| `[contest-CUDA T4]` | `fed97266e88d96d03f1e6780357fa9655c5374fd1a8488335c57f362f0021c38` | 178504 | **0.22620136552710735** | 0.20533002902019143 (PR106 format0d sha `9cb989cef519...`) | **+0.02087** | WORSE — NOT PR111 candidate |
| `[contest-CPU]` linux_x86_64 | `fed97266e88d96d03f1e6780357fa9655c5374fd1a8488335c57f362f0021c38` | 178504 | **0.192042660714715** | 0.192028282957... (DQS1 rank021 sha `7a0da5d0fc327cba...`) | **+0.0000144** (1.4e-5; within rate-axis savings noise) | WORSE — NOT PR111 candidate |

**Per-component breakdown**:

| Component | CUDA | CPU |
|---|---|---|
| avg_segnet_dist | 0.00066299 | 0.00056029 |
| avg_posenet_dist | 0.00016846 | 0.00002943 |
| rate_unscaled | 0.004754339462724803 | 0.004754339462724803 |
| score_seg_contribution | 0.066299 | 0.05602900 |
| score_pose_contribution | 0.04104388 | 0.01715517 |
| score_rate_contribution | 0.11885848656812006 | 0.11885848656812006 |
| score_recomputed | 0.22620136552710735 | 0.192042660714715 |

**Dispatch metadata** (canonical Modal call_id ledger `.omx/state/modal_call_id_ledger.jsonl`):

| Axis | Modal call_id | Status | Elapsed | Cost (approx) |
|---|---|---|---|---|
| CUDA T4 | `fc-01KSKK6DQKB9YEBHR550Y4KB0W` | harvested | 76.3s | $0.025 |
| CPU | `fc-01KSKK713B02QD5NEP5V6FHQTQ` | harvested | 260.8s | $0.01 |
| **First-round CUDA** (failed) | `fc-01KSKJZ91M59T6P1NESGD3QE4R` | failed (stale wrapper) | 5.1s | $0.005 |
| **First-round CPU** (failed) | `fc-01KSKJZY0HR3Z7MEDA3N3RHAAD` | failed (stale wrapper) | 5.3s | $0.005 |

**Total Modal spend**: ~$0.045 (well within $2.00 session budget).

---

## Stacked archive build

**Source**: PR101+FEC6 baseline archive `experiments/results/lane_master_gradient_fec6_modal_t4_cuda_anchor_dispatch_20260520T172434Z_modal/harvested_artifacts/archive.zip` (sha `6bae0201fb082457...`, 178517 bytes).

**Wrapper structure** (`x` member, 178417 bytes):
- 4 bytes outer magic `b"FP11"`
- 4 bytes source_len uint32 = 178158
- source_payload[178158 bytes] (PR101 HNeRV state-dict + masks + poses)
- 2 bytes selector_len uint16 = **249** (FEC6) → **236** (FEC10 hybrid) ← **THE SWAP**
- selector_payload[selector_len bytes] = FEC10 hybrid adaptive-blend at offset 178168

**Bug class anchor**: First-round dispatches `fc-01KSKJZ91M59T6P1NESGD3QE4R` (CUDA) + `fc-01KSKJZY0HR3Z7MEDA3N3RHAAD` (CPU) returned rc=1 after 5s each because the initial swap left the PR101 wrapper's `selector_len` header pointing to 249B while the selector was 236B → `parse_pr101_frame_selector_archive` rejected with `ValueError: FES1 selector payload truncated`. The fix updated the `selector_len` header to 236 (canonical uint16 little-endian); rebuilt archive sha changed from `64c743578b7c2438...` → `fed97266e88d96d0...`. Both V2 dispatches passed inflate + full auth eval clean.

**New `x` member**: 178404 bytes. **Final archive.zip**: 178504 bytes (delta -13 vs FEC6 baseline 178517).

**Deterministic ZIP**: `zipfile.ZipInfo` with fixed timestamp `(2026, 5, 26, 0, 0, 0)`, `ZIP_STORED`, Unix `create_system=3`, mode `0o644 << 16`. Reproducible byte-identical on any Python 3.x.

**Stacked submission_dir tree** (post-fix):

```
experiments/results/pr101_frame_exploit_selector_fec10_hybrid_stacked_20260526T020000Z/submission_dir/
├── README.md          (2.2K — lineage + innovation + HNeRV parity per layer)
├── inflate.sh         (818B — canonical 3-arg contract per Catalog #146)
├── inflate.py         (17.0K / 402 LOC — FEC5/6/8/10 dispatch via magic; FEC10 added per V14)
├── archive.zip        (178504 bytes; sha fed97266e88d96d03f1e6780357fa9655c5374fd1a8488335c57f362f0021c38)
├── encoder/
│   ├── build_pr101_frame_exploit_selector_packet_markov.py  (FEC8 + transition tables)
│   └── build_pr101_frame_exploit_selector_packet_fec10_hybrid.py  (340 LOC FEC10 hybrid)
└── src/
    ├── codec.py
    ├── fec8_markov_decoder.py
    ├── fec10_hybrid_decoder.py  (NEW; 30 LOC re-export shim — single source of truth)
    ├── frame_selector.py
    └── model.py
```

---

## Canonical-vs-unique decision per layer (Catalog #290)

- **PR101 outer wrapper**: ADOPT_CANONICAL (FP11 magic + source/selector framing unchanged from PR101 reference)
- **FEC10 hybrid encoder/decoder**: FORK_BECAUSE_PRINCIPLED_MISMATCH (NEW sister codec to FEC8 family; first to use adaptive-blend Markov context coder per PPM blending pattern; canonical equation #344 entry registered today)
- **FEC10 decoder shim** (`fec10_hybrid_decoder.py`): ADOPT_CANONICAL (sister to `fec8_markov_decoder.py` re-export pattern; single source of truth so encoder + decoder cannot drift)
- **Inflate runtime dispatcher**: ADOPT_CANONICAL (extended existing FEC6/FEC8 magic-dispatch in `unpack_compact_selector_codes` + `unpack_pr101_selector`; <10 LOC delta to add FECa magic)
- **Modal paired-axis dispatch**: ADOPT_CANONICAL (`tools/dispatch_modal_paired_auth_eval.py` + `--skip-axis-if-promotable-anchor-exists` per Catalog #246)
- **Modal call_id ledger**: ADOPT_CANONICAL (`tac.deploy.modal.call_id_ledger.update_call_id_outcome` per Catalog #245)
- **Canonical equation update**: ADOPT_CANONICAL (`tac.canonical_equations.update_equation_with_empirical_anchor` per Catalog #344)
- **Deterministic ZIP repack**: ADOPT_CANONICAL (fixed `date_time`, ZIP_STORED, Unix permissions per Catalog #19 sister discipline)

---

## 9-dimension success checklist evidence (Catalog #294)

1. **UNIQUENESS**: NEW codec sister to FEC8 family (FEC10 adaptive-blend); first canonical equation in registry to PROMOTE from predicted-only to paired contest-CUDA + contest-CPU empirically-validated
2. **BEAUTY+ELEGANCE**: ~80 LOC decoder body; <10 LOC delta to extend `unpack_compact_selector_codes` dispatcher
3. **DISTINCTNESS**: Explicitly different from FEC7 (0-order arith) / FEC8 (per-context Markov) — blends with row-sum-derived weight
4. **RIGOR**: Premise verification via `unpack_fec6_fixed_huffman_codes(x_data[178168:][6:], n_pairs=600)` returning 600 codes BEFORE encoding; roundtrip-verified on real FEC6 stream (`encode → decode → equals codes`)
5. **OPTIMIZATION**: Adaptive-blend ALPHA=2 = empirical optimum across {1,2,4,8,16,32} per RECOVERY-1 commit `39c76755b`
6. **STACK-OF-STACKS**: Sister codec slot on PR101 frame-exploit selector pipeline; same archive grammar; FEC10 magic interleaves with FEC6/FEC8 magic-dispatch table cleanly
7. **DETERMINISTIC REPRODUCIBILITY**: Deterministic ZIP repack + roundtrip-verified encoder/decoder + fixed empirical row-sums baked into static FEC8 2nd-order count tables (zero side-info transmit)
8. **EXTREME OPTIMIZATION**: Total Modal spend $0.045 vs estimated $0.50-1.00 envelope (margin 11-22x under budget); paired dispatch via canonical `--skip-axis-if-promotable-anchor-exists` (no duplicate spend for prior anchors)
9. **OPTIMAL MINIMAL CONTEST SCORE**: -13B archive bytes = -8.66e-6 contest_score_units rate-axis savings (validated empirically on BOTH CPU + CUDA via identical `rate_unscaled = 0.004754339462724803`)

---

## Cargo-cult audit per assumption (Catalog #303)

- **ASSUMPTION 1: "FEC10 selector swap on FEC6 baseline yields PR111 candidate IF rate-axis savings exceed frontier-gap noise"** → CARGO-CULTED. Empirical: rate-axis savings -8.66e-6, CPU frontier gap +1.4e-5 (≈1.6x larger than savings). The FEC6 baseline's distortion components were NEVER measured against DQS1 frontier; they were assumed to be ≤ frontier. Falsified by paired contest-CPU on V14 archive (score 0.192042660714715 vs frontier 0.192028282957 = +1.4e-5 worse).
- **ASSUMPTION 2: "FEC10 hybrid adaptive-blend codec is PARADIGM-VALIDATED via -13B wire-byte savings"** → HARD-EARNED. Empirical: -13B wire savings carried through Modal contest auth-eval pipeline on BOTH CPU + CUDA axes (rate_unscaled identical to predicted 0.004754339462724803; archive_size_bytes = 178504 = 178517 - 13 - 0 ZIP framing delta). Roundtrip-verified end-to-end via real FEC6 stream.
- **ASSUMPTION 3: "FEC6 baseline (sha `6bae0201`) is the right substrate to stack FEC10 on"** → CARGO-CULTED (suppression-anti-pattern per UNIQUE-AND-COMPLETE-PER-METHOD). The DQS1 frontier (sha `7a0da5d0...`) and PR106 format0d frontier (sha `9cb989cef519...`) use DIFFERENT selector approaches; their selector packet structure may not exist or may differ. The FEC10 hybrid codec library SHOULD be re-evaluated against DQS1 / PR106 selector stacks (if compatible) OR against future class-shift substrates that beat DQS1. Reactivation criterion: identify DQS1 / PR106 / next-class-shift substrate's selector packet; verify selector codes are 16-symbol palette; re-stack + re-paired-dispatch.
- **ASSUMPTION 4: "Canonical equation #344 entry needs PROMOTION via paired empirical anchors"** → HARD-EARNED. PROMOTION landed: equation has 3 anchors (1 predicted + 1 contest-CUDA + 1 contest-CPU; all residual=1.0 since predicted output exactly matches empirical output on rate axis); registry total 55 equations.

---

## Observability surface (Catalog #305)

- **Inspectable per layer**: archive.zip (`zipfile.ZipFile` + member `x` extraction); PR101 wrapper (`parse_pr101_frame_selector_archive`); selector payload (`unpack_compact_selector_codes` dispatched by magic); FEC10 decoder (`_decode_adaptive_blend` per-pair codelen via `_codelen_per_pair_blend` helper)
- **Decomposable per signal**: contest_auth_eval.json carries `score_seg_contribution` + `score_pose_contribution` + `score_rate_contribution` separately on both axes; `rate_unscaled` validates exact byte-count → rate-axis projection
- **Diff-able across runs**: Modal call_id ledger appends per-axis `harvested` events; `dispatch_modal_paired_auth_eval.py` plan JSON at `experiments/results/.../v14_paired_modal_execute_plan_v2.json` captures full command + expected runtime tree sha for both axes
- **Queryable post-hoc**: `tac.canonical_equations.query_equations()` returns the equation with all 3 anchors keyed by `equation_id`; `.omx/state/modal_call_id_ledger.jsonl` queryable via `query_by_call_id` / `query_by_lane`; `.omx/state/active_lane_dispatch_claims.md` queryable via `tools/claim_lane_dispatch.py summary`
- **Cite-able**: every anchor carries canonical `Provenance` per Catalog #323 (archive_zip_path + contest_archive_member_name + measurement_axis + hardware_substrate + evidence_grade + captured_at_utc + score_claim_valid + promotion_eligible=False since neither axis beats frontier)
- **Counterfactual-able**: FEC10 hybrid encoder accepts arbitrary `codes` lists + `alpha` parameter sweep; can re-encode FEC6 baseline with α ∈ {1,2,4,8,16,32} to verify ALPHA=2 optimum; can re-encode DQS1 frontier's selector codes if compatible (open follow-up)

---

## Predicted ΔS band (Catalog #296 Dykstra-feasibility check)

**Predicted ΔS** (pre-dispatch): -8.7e-6 contest_score_units (closed-form from `Δrate = 25 * (178504 - 178517) / 37545489`). Per Shannon R(D) bound + the BEFORE-coder context-model enhancement paradigm: the rate-axis savings is achievable IFF (a) FEC10 decoder is byte-identical to encoder (roundtrip-verified), AND (b) inflate runtime preserves all OTHER bytes (PR101 source_payload + PR101 wrapper headers verified bytewise identical).

**Empirical ΔS** (paired-axis): The rate-axis savings -8.66e-6 is preserved EXACTLY on both axes (rate_unscaled = 0.004754339462724803 on both CPU + CUDA, identical to predicted). However, the FEC6 baseline's distortion components were NOT in the predicted band; the canonical equation #344 entry's `domain_of_validity.in_domain_contexts` is ONLY `selector_index_stream_pr101_frame_exploit_600_pairs_k16_palette` — it predicts wire-byte savings (validated) but does NOT predict total-score positioning vs frontier.

**Dykstra-feasibility implication**: The achievable Pareto frontier for the (rate, seg, pose) polytope is determined by the underlying renderer/codec architecture, NOT by the selector packet alone. Saving -13B on selector bytes is necessary-but-not-sufficient for frontier-pursuit; the FEC6 baseline's renderer+codec architecture sits ABOVE the DQS1 / PR106 frontiers on both axes. Per CLAUDE.md "Meta-Lagrangian/Pareto solver": "Saving rate bytes on a substrate above the frontier does NOT shift the achievable Pareto frontier unless distortion components are also improved."

---

## 6-hook wire-in declaration (Catalog #125)

1. **Sensitivity-map contribution**: ACTIVE. The contest_auth_eval.json per-component breakdown (seg / pose / rate) feeds `tac.sensitivity_map.*` consumers as per-axis sensitivity rows. Rate-axis savings empirically validated (Δrate = -8.66e-6 on both axes).
2. **Pareto constraint**: ACTIVE. New paired (CPU, CUDA) anchor row on the achievable Pareto polytope. Anchor verifies FEC10 selector swap shifts archive_bytes axis but does NOT shift the (seg, pose) projection meaningfully (Δseg ≈ 0, Δpose ≈ 0 vs FEC6 baseline).
3. **Bit-allocator hook**: ACTIVE. The FEC10 hybrid encoder is registered as canonical producer of `cascade_a_fec10_hybrid_adaptive_blend_savings_v1` (per `tac.canonical_equations`); bit-allocator can call `encode_fec10_hybrid_adaptive_blend(codes, n_pairs=N)` to get optimal bit-count for any selector code list ∈ {16-palette, 100-10000 pairs}.
4. **Cathedral autopilot dispatch hook**: ACTIVE. Both Modal calls auto-registered in canonical `.omx/state/modal_call_id_ledger.jsonl` (per Catalog #245). Outcome events landed; cathedral autopilot ranker can re-query via `tac.deploy.modal.call_id_ledger.latest_status_by_call_id` for any future re-deliberation.
5. **Continual-learning posterior update**: ACTIVE. Canonical equation #344 entry PROMOTED from 1 anchor (predicted-only) → 3 anchors (predicted + contest-CUDA + contest-CPU). `tac.canonical_equations.update_equation_with_empirical_anchor` appended via fcntl-locked JSONL per Catalog #131. Auto-recalibration triggers when 3+ new empirical anchors land in domain.
6. **Probe-disambiguator**: ACTIVE. The paired CPU+CUDA anchor IS the canonical disambiguator between (a) "FEC10 hybrid codec works in principle" (validated; -13B wire savings preserved) and (b) "FEC10 hybrid stacked archive IS PR111 candidate" (FALSIFIED; both axes WORSE than frontier). Future probe: re-stack FEC10 onto DQS1 / PR106 / class-shift substrate IF selector packet structure compatible.

---

## Operator-routable next steps

1. **PR111 NOT-CANDIDATE** (V14 verdict). FEC10 hybrid stacked on PR101+FEC6 baseline is WORSE than canonical frontier on both axes. Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA": no PR submission via canonical pipeline.

2. **FEC10 hybrid sister codec library is AVAILABLE** for future composition. Reactivation criteria:
   - **Path A**: Identify DQS1 frontier (sha `7a0da5d0...`) or PR106 format0d frontier (sha `9cb989cef519...`) selector packet structure; verify 16-symbol palette + 600-pair format compatible; re-stack FEC10 + re-paired-dispatch.
   - **Path B**: When next class-shift substrate (per Catalog #233 L1→L2 promotion) lands with a PR101-compatible selector packet, compose FEC10 hybrid + class-shift substrate; estimate composite ΔS via canonical equation #344 entry + class-shift substrate's predicted band.
   - **Path C**: Catalog #325 per-substrate symposium re-deliberation on FEC10 hybrid composition with FEC6 baseline vs frontier substrates — operator decides whether the +1.4e-5 CPU gap is recoverable via further selector-stream entropy reduction (e.g. FEC11 = larger context window, FEC12 = blended-with-3rd-order, etc.) on the FEC6 baseline OR is structurally swamped by FEC6 baseline distortion components.

3. **Canonical equation #344 entry status**: PROMOTED from `predicted_only` to `predicted_plus_contest_cuda_plus_contest_cpu` (3 anchors total). The next anchor that triggers recalibration is whichever new empirical anchor lands first (per `RECALIBRATE_ON_NEW_ANCHORS` trigger).

4. **Canonical frontier pointer**: NOT updated by this subagent (per Catalog #343 operator-routable discipline). Pointer remains at `0.192028282957 [contest-CPU]` + `0.205330029020 [contest-CUDA]`. Operator decision required if Catalog #343 frontier-band update needed for the within-noise V14 CPU anchor (not recommended; FEC10 stacked is empirically WORSE).

5. **Lane registry**: `lane_v14_cascade_a_fec10_hybrid_p11_paired_cpu_cuda_pr111_candidate_20260526` marked `completed_paired_cpu_cuda_NOT_PR111_CANDIDATE` in `.omx/state/active_lane_dispatch_claims.md` with full custody. Sister lane `lane_v14_cascade_a_fec10_hybrid_p11_paired_cpu_cuda_pr111_candidate_v2_20260526_contest_{cuda,cpu}` carries the per-axis terminal verdict.

---

## Cross-references

- RECOVERY-1 commit `39c76755b` — canonical equation registry 52 → 53 (FEC10 hybrid registered as predicted-only)
- This subagent: canonical equation registry 53 → 55 (NOT counting the FEC10 entry which was PROMOTED, not added; the +2 reflects today's session-cumulative growth from CASCADE B CATALYST + sister Cascade A FEC8 anchors that landed in the interim per `tac.canonical_equations.query_equations()` returning 55 total at landing)
- Canonical equation: `cascade_a_fec10_hybrid_adaptive_blend_savings_v1` (entry in `.omx/state/canonical_equations_registry.jsonl`)
- Stacked archive: `experiments/results/pr101_frame_exploit_selector_fec10_hybrid_stacked_20260526T020000Z/submission_dir/archive.zip`
- Paired Modal plans:
  - `experiments/results/pr101_frame_exploit_selector_fec10_hybrid_stacked_20260526T020000Z/v14_paired_modal_plan.json` (first round, failed)
  - `experiments/results/pr101_frame_exploit_selector_fec10_hybrid_stacked_20260526T020000Z/v14_paired_modal_execute_plan_v2.json` (second round, succeeded)
- Source canonical equation registration script: `tools/register_cascade_a_fec10_hybrid_adaptive_blend_canonical_equation_20260526.py` (RECOVERY-1)
- Canonical frontier pointer: `.omx/state/canonical_frontier_pointer.json` (refresh: `.venv/bin/python tools/refresh_canonical_frontier.py`)

---

## CLAUDE.md non-negotiable compliance

- ✅ **Modal `.spawn()` HARVEST OR LOSE**: both call_ids registered in canonical ledger + outcomes recorded via `update_call_id_outcome`
- ✅ **Apples-to-apples evidence discipline**: paired CPU + CUDA on SAME archive bytes; axis tags explicit; hardware substrates declared
- ✅ **Submission auth eval — BOTH CPU AND CUDA**: explicit verdict that neither axis beats frontier; NO PR submission
- ✅ **eval_roundtrip + EMA**: inherited from PR101+FEC6 baseline; no architectural change
- ✅ **MPS auth eval is NOISE**: paired dispatch via Modal CPU + Modal T4 (1:1 contest-compliant hardware per CLAUDE.md); ZERO MPS
- ✅ **Forbidden /tmp paths**: stacked archive + landing memo + canonical equation update all at canonical `experiments/results/` + `.omx/research/` + `.omx/state/` paths
- ✅ **Forbidden component-aliasing**: archive sha verified pre + post paired dispatch
- ✅ **NEVER invent CLI flags**: `tools/dispatch_modal_paired_auth_eval.py` invoked with verified argparse signature
- ✅ **Forbidden score claims**: every score literal in this memo is tagged with axis (`[contest-CPU]`, `[contest-CUDA T4]`) AND tagged via canonical Provenance per Catalog #287/#323 (the entire content is the per-anchor empirical_output dict, NOT a hardcoded literal — Catalog #343 sister discipline)
- ✅ **Strict-flip atomicity**: no new STRICT preflight gates added (per Catalog #299 quota brake; current ≈360 well under 400)
- ✅ **Subagent coherence-by-default**: read CLAUDE.md + RECOVERY-1 commit + canonical frontier pointer + canonical equation registry BEFORE any edit (Catalog #229 PV)
- ✅ **Catalog #117/#157/#174/#235/#289 commit serializer discipline**: all commits via `tools/subagent_commit_serializer.py` + POST-EDIT `--expected-content-sha256`
- ✅ **Catalog #119 Co-Authored-By Claude trailer**: included in all subagent commits
- ✅ **Catalog #206 checkpoint discipline**: 7+ checkpoints emitted to `.omx/state/subagent_progress.jsonl`
- ✅ **Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE**: canonical equation update is APPEND-ONLY (new anchors added; existing predicted anchor preserved verbatim)
- ✅ **Catalog #230 sister-disjoint scope**: stayed within stacked-archive build + paired-CUDA dispatch + canonical equation registration + landing memo + lane registry; NOT touching Phase 2 / WAVE-5 (sister-disjoint scopes per prompt)
- ✅ **Catalog #340 sister-checkpoint guard**: PROCEED (no sister conflicts at edit-time)
- ✅ **Catalog #344 PROMOTION discipline**: canonical equation #344 entry PROMOTED with paired empirical CUDA + CPU anchors only (per CLAUDE.md "PROMOTION only with paired-axis empirical evidence")

---

## Mission contribution per Catalog #300

`frontier_protecting` — extincts the implicit assumption that "selector-stream rate savings are sufficient for PR submission" by paired-axis empirical falsification. The canonical equation #344 entry's `domain_of_validity` is now empirically constrained: predicts WIRE-BYTE savings (validated) but NOT total-score positioning vs frontier (must check Pareto-feasibility separately). FEC10 hybrid sister codec library remains AVAILABLE for future composition with class-shift substrates per Path A/B/C above.

Per CLAUDE.md "Forbidden premature KILL without research exhaustion": this is DEFERRED-pending-research (Paths A+B+C above are valid reactivation criteria), NOT KILLED. The codec PARADIGM is validated (-13B wire savings preserved across paired axes); the IMPLEMENTATION-LEVEL outcome (NOT PR111 candidate when stacked on FEC6 baseline) is a substrate-mismatch result per Catalog #307 paradigm-vs-implementation classification — sister anchor of the existing FEC10 falsification of P13+P15 sub-components from RECOVERY-1.
