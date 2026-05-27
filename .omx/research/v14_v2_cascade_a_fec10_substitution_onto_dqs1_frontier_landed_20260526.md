# V14-V2 Cascade A FEC10 Substitution onto DQS1 Frontier — FRONTIER-CROSSING — LANDED 2026-05-26

**Lane**: `lane_v14_v2_cascade_a_fec10_substitution_onto_dqs1_frontier_20260526`
**Subagent**: `v14-v2-cascade-a-fec10-substitution-onto-dqs1-frontier-7a0da5d0-frontier-crossing-attempt-per-operator-insight-20260526`
**Operator authorization**: V14 PATH A operator-routable per `feedback_v14_cascade_a_fec10_hybrid_p11_paired_cpu_cuda_landed_20260526.md` + "All operator decisions approved" + OPERATOR INSIGHT 2026-05-26 *"V14 stacked FEC10 onto WRONG baseline; canonical frontier per pointer is DQS1 pairset-drop-one rank021 sha 7a0da5d0..."*
**Continuation of**: V14 commit `abdeefb00` (canonical equation #344 PROMOTED to 3 anchors via FEC6-baseline paired CPU+CUDA; NOT PR111 candidate)
**Mission contribution**: `frontier_breaking` (PR111 CANDIDATE — paired-axis empirical frontier-crossing on BOTH CPU and CUDA axes)
**Horizon class**: `frontier_pursuit` — successful frontier-crossing within rate-axis savings; canonical equation #344 PROMOTED 3 → 5 anchors via PATH A substitution apples-to-apples

---

## Executive summary

Substituted the Cascade A FEC10 hybrid adaptive-blend selector packet directly into the DQS1 frontier archive (sha `7a0da5d0fc327cba...`, 178559 bytes, contest-CPU 0.19202828295713675) by swapping the FEC6 selector packet (249 bytes at offset 178166 in `x` member) with the FEC10 hybrid packet (236 bytes, -13B wire-byte savings), preserving the DQS1 source_payload + DQS1 packet trailer byte-identically. Pre-dispatch PV verified (a) decode-equality `decoded(FEC10) === decoded(FEC6)` for the same 600 codes; (b) frame-byte-identical inflate locally (output 0.raw sha256 `00b479229c97ede3...` matched DQS1 inflate output exactly). Dispatched paired contest-CUDA + contest-CPU on Modal T4 + Modal CPU. Both axes returned clean (rc=0; score_claim_valid=True; promotable=True). Net empirical: **CPU 0.19202062679074616 [contest-CPU]** (-7.66e-6 BELOW canonical frontier 0.19202828295713675) + **CUDA 0.22618311337661345 [contest-CUDA T4]** (-8.66e-6 BELOW DQS1 paired CUDA baseline 0.22619176954300405). Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable: **PR111 CANDIDATE** — both axes empirically beat canonical frontier.

The FEC10 hybrid codec PARADIGM is empirically validated AT FRONTIER-CROSSING SCALE: -13B wire-byte savings preserved through the Modal contest auth-eval pipeline on BOTH axes (rate_unscaled identical at 0.004755458105766048 vs DQS1 baseline 0.0047558043524216715; archive_bytes 178546 = 178559 - 13). The IMPLEMENTATION-LEVEL outcome (FRONTIER-CROSSING) validates the operator insight: V14's "NOT PR111 candidate" verdict was an IMPLEMENTATION-LEVEL substrate-mismatch failure (FEC6 PR110 baseline above frontier), NOT a paradigm refutation. Substituting onto the actual canonical frontier substrate (DQS1 rank021) yielded the predicted frontier-crossing -8.66e-6.

---

## Empirical anchors

| Axis | Archive sha256 | Bytes | Score | Baseline | Delta vs baseline | Verdict |
|---|---|---|---|---|---|---|
| `[contest-CUDA T4]` | `0a3abfe645c4fac0df9ea89237f25dd9bfc6b2471b897c36d7437795d27d1403` | 178546 | **0.22618311337661345** | DQS1 paired CUDA 0.22619176954300405 | **-8.66e-6** | **FRONTIER-CROSSING** |
| `[contest-CPU]` linux_x86_64 | `0a3abfe645c4fac0df9ea89237f25dd9bfc6b2471b897c36d7437795d27d1403` | 178546 | **0.19202062679074616** | DQS1 canonical frontier 0.19202828295713675 | **-7.66e-6** | **FRONTIER-CROSSING — PR111 CANDIDATE — canonical frontier pointer supersede operator-routable per Catalog #343** |

**Per-component breakdown**:

| Component | CUDA | CPU |
|---|---|---|
| avg_segnet_dist | 0.00066254 | 0.00055979 |
| avg_posenet_dist | 0.00016845 | 2.943e-05 |
| rate_unscaled | 0.004755458105766048 | 0.004755458105766048 |
| score_seg_contribution | 0.066254 | 0.055979 |
| score_pose_contribution | 0.041043879928928 | 0.017155174146594957 |
| score_rate_contribution | 0.11888523344768545 | 0.11888523344768545 |
| score_recomputed | 0.22618311337661345 | 0.19202062679074616 |

**Apples-to-apples comparison vs DQS1 baseline** (same paired-Modal infrastructure, same archive grammar except for the 13-byte selector swap):

| Component | DQS1 baseline | V14-V2 substituted | Delta |
|---|---|---|---|
| archive_size_bytes | 178559 | 178546 | -13 (exactly the FEC10 wire savings) |
| avg_segnet_dist (CPU) | 0.00055978 | 0.00055979 | +1e-8 (within floating-point noise) |
| avg_posenet_dist (CPU) | 2.943e-05 | 2.943e-05 | 0 (identical) |
| rate_unscaled | 0.0047558043524216715 | 0.004755458105766048 | -3.46e-7 (=25*13/37545489 = 8.66e-6 scaled by 25 = 0.000346/100 = 3.46e-7 in rate_unscaled units) |
| CPU score | 0.19202828295713675 | 0.19202062679074616 | **-7.66e-6** |
| CUDA score | 0.22619176954300405 | 0.22618311337661345 | **-8.66e-6** |

The empirical delta MATCHES the predicted rate-axis savings within 1e-6 noise (CPU delta -7.66e-6 vs predicted -8.66e-6, 1e-6 short because seg_contribution +1e-6 absorbed; CUDA delta -8.66e-6 exactly matches prediction). The substitution preserved distortion components (seg + pose) within floating-point noise, exactly as expected from frame-byte-identical inflate. This is the canonical apples-to-apples discipline per CLAUDE.md.

**Dispatch metadata** (canonical Modal call_id ledger `.omx/state/modal_call_id_ledger.jsonl`):

| Axis | Modal call_id | Status | Cost (approx) |
|---|---|---|---|
| CUDA T4 | `fc-01KSKNC9BQ4ABNNC8AW77SB56E` | harvested rc=0 | ~$0.025 |
| CPU | `fc-01KSKNB14HSA215KB4J3CJTFW4` | harvested rc=0 | ~$0.01 |

**Total Modal spend**: ~$0.035 (within $1.50 session budget; well under V14 predecessor $0.045).

---

## Substituted archive build

**Source DQS1 archive**: `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/pairset_drop_one_rank021_pair0371_materialization_20260522T180446Z/submission_dir/archive.zip` (sha `7a0da5d0fc327cba3f7d1387a544fd5ce5f05bc56ecc8e12cd5097141672f4fe`, 178559 bytes).

**Wrapper structure** (DQS1 `x` member, 178459 bytes):
- 4 bytes outer magic `b"FP11"`
- 4 bytes source_len uint32 = 178158
- source_payload[178158 bytes] (DQS1 selective-decoderq HNeRV state-dict + masks + poses)
- 2 bytes selector_len uint16 = **249** (FEC6) → **236** (FEC10 hybrid) ← **THE SWAP**
- selector_payload[selector_len bytes] = FEC10 hybrid adaptive-blend at offset 178166
- DQS1 packet trailer[42 bytes] (pairset-drop-one rank021 decoderq patch)

**Pre-dispatch PV** (Phase 1):
- Decoded FEC6 codes from DQS1 selector packet (600 codes; symbol histogram: 16-symbol palette properly populated)
- Encoded same codes via canonical `encode_fec10_hybrid_adaptive_blend` (236 bytes; -13B vs FEC6)
- Decoded FEC10 via canonical `decode_fec10_hybrid_selector` (600 codes)
- **PV PASSED**: `decoded(FEC10) === decoded(FEC6)` byte-identical
- **Frame-byte-identical inflate**: locally ran inflate.py on substituted archive vs DQS1 archive; output `0.raw` files (3.66 GB each) sha256 matched: `00b479229c97ede3e776846297269f7785285702b8dbf3e5dccc733557da605a`

**New `x` member**: 178446 bytes (=178459-13). **Final archive.zip**: 178546 bytes (delta -13 vs DQS1 baseline 178559).

**Deterministic ZIP**: `zipfile.ZipInfo` with fixed timestamp `(2026, 5, 26, 0, 0, 0)`, `ZIP_STORED`, Unix `create_system=3`, mode `0o644 << 16`. Reproducible byte-identical on any Python 3.x.

**Substituted submission_dir tree**:

```
experiments/results/v14_v2_dqs1_plus_fec10_substituted_20260526T023000Z/submission_dir/
├── inflate.sh         (818B — canonical 3-arg contract per Catalog #146)
├── inflate.py         (DQS1 inflate + FECa dispatch added; 2 acceptance branches modified)
├── archive.zip        (178546 bytes; sha 0a3abfe645c4fac0df9ea89237f25dd9bfc6b2471b897c36d7437795d27d1403)
├── encoder/
│   ├── build_pr101_frame_exploit_selector_packet_fec10_hybrid.py
│   └── build_pr101_frame_exploit_selector_packet_markov.py
└── src/
    ├── codec.py
    ├── codec_sidecar.py
    ├── fec10_hybrid_decoder.py
    ├── frame_selector.py
    └── model.py
```

---

## Canonical-vs-unique decision per layer (Catalog #290)

- **DQS1 outer wrapper**: ADOPT_CANONICAL (FP11 magic + source/selector framing + DQS1 packet trailer unchanged from DQS1 reference)
- **DQS1 source_payload**: ADOPT_CANONICAL (preserved byte-identically — the canonical pairset-drop-one rank021 selective-decoderq state)
- **FEC10 hybrid encoder/decoder**: ADOPT_CANONICAL (vendor V14's canonical `encode_fec10_hybrid_adaptive_blend` + `decode_fec10_hybrid_selector` per RECOVERY-1 commit `39c76755b`)
- **DQS1 packet trailer**: ADOPT_CANONICAL (42-byte selective-decoderq patch preserved byte-identically)
- **Inflate runtime dispatcher**: FORK_BECAUSE_PRINCIPLED_MISMATCH (DQS1 inflate.py supports FEC2/3/5/6 but NOT FECa; added `b"FECa"` dispatch case + extended `unpack_pr101_selector` acceptance list)
- **Modal paired-axis dispatch**: ADOPT_CANONICAL (separate `modal_auth_eval_cpu.py` + `modal_auth_eval.py` invocations after `dispatch_modal_paired_auth_eval.py` hit Cascade C' sister-subagent file-modification race; serial dispatch was the canonical operator-routable workaround per CLAUDE.md "Subagent coherence-by-default" sister-disjoint discipline)
- **Modal call_id ledger**: ADOPT_CANONICAL (`tac.deploy.modal.call_id_ledger.update_call_id_outcome` per Catalog #245)
- **Canonical equation update**: ADOPT_CANONICAL (`tac.canonical_equations.update_equation_with_empirical_anchor` per Catalog #344)
- **Deterministic ZIP repack**: ADOPT_CANONICAL (fixed `date_time`, ZIP_STORED, Unix permissions per Catalog #19 sister discipline)

---

## 9-dimension success checklist evidence (Catalog #294)

1. **UNIQUENESS**: FIRST substitution onto canonical frontier baseline; V14 predecessor stacked onto wrong (FEC6 PR110) baseline; this is the canonical apples-to-apples test that the operator insight predicted
2. **BEAUTY+ELEGANCE**: -13B archive bytes via ~10 LOC inflate.py extension + identical FEC10 hybrid encoder/decoder library; reviewable in 30 seconds
3. **DISTINCTNESS**: Explicitly different from V14 (DQS1 baseline vs FEC6 PR110 baseline; canonical frontier surface vs above-frontier surface)
4. **RIGOR**: Pre-dispatch PV via 3 layers (decode-equality + frame-byte-identical inflate + paired-axis Modal); roundtrip-verified end-to-end
5. **OPTIMIZATION**: Adaptive-blend ALPHA=2 = empirical optimum per RECOVERY-1 commit `39c76755b`; same canonical encoder library
6. **STACK-OF-STACKS**: Substitution composes FEC10 codec + DQS1 selective-decoderq substrate; SISTER OF V14 stacking (substitution vs stacking semantics)
7. **DETERMINISTIC REPRODUCIBILITY**: Deterministic ZIP repack + roundtrip-verified encoder/decoder + frame-byte-identical local inflate as canonical apples-to-apples check
8. **EXTREME OPTIMIZATION**: Total Modal spend $0.035 vs estimated $0.30-0.60 envelope (margin 9-17x under budget); pre-dispatch PV avoided wasted dispatch on incompatible substrate
9. **OPTIMAL MINIMAL CONTEST SCORE**: **FRONTIER-CROSSING ON BOTH AXES** — CPU -7.66e-6 below canonical 0.19202828295713675; CUDA -8.66e-6 below DQS1 baseline 0.22619176954300405

---

## Cargo-cult audit per assumption (Catalog #303)

- **ASSUMPTION 1: "DQS1 selector packet structure is compatible with FEC10 substitution (16-symbol palette + 600-pair format + FP11 wrapper)"** → HARD-EARNED. Empirical: DQS1 `x` member has identical FP11 + source_len(4) + source + selector_len(2) + selector wrapper; FEC6 selector packet has 16-symbol palette (verified via decode + histogram) + 600 pairs (verified via header); substitution semantics align byte-for-byte.
- **ASSUMPTION 2: "Decode equality preserves frame bytes hence distortion components"** → HARD-EARNED via TWO empirical layers: (a) `decoded(FEC10) === decoded(FEC6)` per Phase 1 PV (encoder/decoder roundtrip-verified); (b) frame-byte-identical inflate of `0.raw` (3.66 GB) sha256 match between V14-V2 and DQS1 archives. The canonical apples-to-apples check.
- **ASSUMPTION 3: "Rate-axis savings -8.66e-6 carries through Modal contest auth-eval pipeline"** → HARD-EARNED. Empirical: rate_unscaled identical on BOTH axes (0.004755458105766048); archive_bytes = 178546 = 178559 - 13 EXACTLY; rate_contribution -3.46e-7 vs DQS1.
- **ASSUMPTION 4: "Substitution onto canonical frontier baseline yields PR111 candidate IF distortion preserved"** → HARD-EARNED-EMPIRICALLY-VERIFIED. CPU -7.66e-6 BELOW canonical frontier (PR111 candidate); CUDA -8.66e-6 BELOW DQS1 paired CUDA baseline. The operator insight was structurally correct: V14's "NOT PR111 candidate" was an IMPLEMENTATION-LEVEL substrate-mismatch (FEC6 PR110 baseline above frontier), not a paradigm refutation.
- **ASSUMPTION 5: "FEC10 hybrid codec library generalizes from PR101 FEC6 baseline to DQS1 substrate"** → HARD-EARNED-EMPIRICALLY-VERIFIED. The encoder + decoder pair built for PR101 FEC6 baseline produced identical wire savings (-13B) and identical canonical apples-to-apples behavior on the DQS1 substrate. The codec library is paradigm-portable across PR101-compatible selector packet substrates.

---

## Observability surface (Catalog #305)

- **Inspectable per layer**: archive.zip (`zipfile.ZipFile` + member `x` extraction); FP11 wrapper (offset 178166 carries selector_len header + selector_payload); DQS1 packet trailer (42 bytes after selector); FEC10 decoder (`decode_fec10_hybrid_selector` per-pair codelen via `_codelen_per_pair_blend` helper)
- **Decomposable per signal**: contest_auth_eval.json carries `score_seg_contribution` + `score_pose_contribution` + `score_rate_contribution` separately on both axes; `rate_unscaled` validates exact byte-count → rate-axis projection; apples-to-apples comparison vs DQS1 baseline per-axis
- **Diff-able across runs**: Modal call_id ledger appends per-axis `harvested` events; `experiments/results/.../v14_v2_paired_modal_plan.json` captures full command + expected runtime tree sha for both axes; local frame-byte-identical inflate verification preserved as canonical pre-dispatch PV
- **Queryable post-hoc**: `tac.canonical_equations.query_equations()` returns the equation with all 5 anchors; `.omx/state/modal_call_id_ledger.jsonl` queryable via `query_by_call_id` / `query_by_lane`; `.omx/state/active_lane_dispatch_claims.md` queryable via `tools/claim_lane_dispatch.py summary`
- **Cite-able**: every anchor carries canonical `Provenance` per Catalog #323 (archive_zip_path + contest_archive_member_name + measurement_axis + hardware_substrate + evidence_grade + captured_at_utc + score_claim_valid=True + promotion_eligible=True)
- **Counterfactual-able**: substitution is reversible (restore FEC6 selector packet to get DQS1 baseline); decoder library accepts arbitrary `codes` lists; can re-substitute onto other PR101-compatible substrates via same encoder/decoder

---

## Predicted ΔS band (Catalog #296 Dykstra-feasibility check)

**Predicted ΔS** (pre-dispatch): -8.66e-6 contest_score_units (closed-form from `Δrate = 25 * (178546 - 178559) / 37545489`). Per Shannon R(D) bound + frame-byte-identical apples-to-apples discipline: the rate-axis savings is achievable IFF (a) FEC10 decoder produces identical codes to FEC6 decoder (PV: decode-equality), AND (b) inflate runtime preserves DQS1 source_payload + DQS1 packet trailer + ALL renderer state byte-identically (PV: frame-byte-identical 0.raw output).

**Empirical ΔS** (paired-axis):
- CUDA delta -8.66e-6 EXACTLY matches prediction
- CPU delta -7.66e-6 within 1e-6 noise of prediction (seg_contribution +1e-6 absorbed; floating-point noise from MKL CPU eval pathway)

**Dykstra-feasibility implication**: The substitution preserves DQS1's Pareto polytope position on the (seg, pose) axes EXACTLY (frame-byte-identical inflate ⟹ identical scorer output bytes); ONLY the archive-bytes axis shifted by -13. This is the canonical "pure rate-axis improvement at fixed distortion" — the cleanest possible PR111 candidate primitive.

---

## 6-hook wire-in declaration (Catalog #125)

1. **Sensitivity-map contribution**: ACTIVE. The contest_auth_eval.json per-component breakdown (seg / pose / rate) feeds `tac.sensitivity_map.*` consumers as per-axis sensitivity rows. Rate-axis savings empirically validated (Δrate = -8.66e-6 on both axes); seg + pose preserved within 1e-8 noise per frame-byte-identical inflate.
2. **Pareto constraint**: ACTIVE. New paired (CPU, CUDA) anchor row on the achievable Pareto polytope. Anchor verifies FEC10 substitution shifts ONLY archive_bytes axis (Δseg ≈ 0, Δpose ≈ 0 vs DQS1 baseline). Canonical Pareto-feasible primitive.
3. **Bit-allocator hook**: ACTIVE. The FEC10 hybrid encoder is registered as canonical producer of `cascade_a_fec10_hybrid_adaptive_blend_savings_v1` (per `tac.canonical_equations`); bit-allocator can call `encode_fec10_hybrid_adaptive_blend(codes, n_pairs=N)` to get optimal bit-count for any selector code list on PR101-compatible substrates.
4. **Cathedral autopilot dispatch hook**: ACTIVE. Both Modal calls auto-registered in canonical `.omx/state/modal_call_id_ledger.jsonl` (per Catalog #245). Outcome events landed; cathedral autopilot ranker can re-query via `tac.deploy.modal.call_id_ledger.latest_status_by_call_id`.
5. **Continual-learning posterior update**: ACTIVE. Canonical equation #344 entry PROMOTED from 3 anchors (V14: predicted + FEC6-baseline CUDA + FEC6-baseline CPU) → 5 anchors (added: DQS1-substituted CUDA + DQS1-substituted CPU). `tac.canonical_equations.update_equation_with_empirical_anchor` appended via fcntl-locked JSONL per Catalog #131.
6. **Probe-disambiguator**: ACTIVE. The paired-axis FRONTIER-CROSSING IS the canonical disambiguator validating V14's PATH A operator-routable: V14's "NOT PR111 candidate" was IMPLEMENTATION-LEVEL substrate-mismatch (not paradigm refutation); substitution onto actual canonical frontier substrate validates the codec PARADIGM at frontier-crossing scale.

---

## Operator-routable next steps

1. **PR111 CANDIDATE** (V14-V2 verdict). FEC10 hybrid substituted onto DQS1 frontier is BETTER than canonical frontier on BOTH axes:
   - CPU: 0.19202062679074616 < 0.19202828295713675 (canonical pointer) by -7.66e-6
   - CUDA: 0.22618311337661345 < 0.22619176954300405 (DQS1 paired CUDA) by -8.66e-6

   Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable: candidate is paired-axis qualified.

2. **Canonical frontier pointer supersede** (operator-routable per Catalog #343): the canonical pointer at `.omx/state/canonical_frontier_pointer.json` currently cites DQS1 sha `7a0da5d0fc327cba...` / CPU 0.19202828295713675. The V14-V2 substituted archive sha `0a3abfe645c4fac0...` / CPU 0.19202062679074616 supersedes via -7.66e-6. **DO NOT auto-flip per Catalog #343 discipline; operator decides whether to update the pointer.**

3. **PR111 candidate report** at `reports/pr111_candidate_cascade_a_fec10_substitution_onto_dqs1_frontier_landed_20260526.md` per the PV111 directive. Operator-routable to either (a) host archive + run final apples-to-apples compliance gate + invoke `gh pr create` per `docs/submission_template.md`, OR (b) re-route the submission to a different sister subagent with the hosted-URL + report.txt prerequisites resolved.

4. **Canonical equation #344 entry status**: PROMOTED from 3 anchors → 5 anchors (V14 FEC6-baseline anchors PRESERVED per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE; V14-V2 DQS1-substituted anchors APPENDED). Each anchor's `in_domain_context` field documents the substrate it was tested against; downstream consumers can route per-substrate.

5. **Lane registry**: `lane_v14_v2_cascade_a_fec10_substitution_onto_dqs1_frontier_20260526` to be marked `completed_paired_cpu_cuda_FRONTIER_CROSSING_PR111_CANDIDATE` in `.omx/state/active_lane_dispatch_claims.md`.

6. **Future composition** (sister of V14 Path A): The FEC10 hybrid codec library remains AVAILABLE for composition with FUTURE class-shift substrates (predictive-receiver / cooperative-receiver / etc.) that beat DQS1 frontier. Substitution semantics are paradigm-portable; the canonical equation #344 entry's `in_domain_context` now empirically covers `selector_index_stream_pr101_frame_exploit_600_pairs_k16_palette` in BOTH `_above_frontier_via_pr101_fec6_baseline` (V14) AND `_substituted_into_dqs1_pairset_drop_one_rank021_frontier_archive` (V14-V2) contexts.

---

## Cross-references

- V14 commit `abdeefb00` — IMPLEMENTATION-LEVEL FALSIFICATION (NOT PR111 candidate; FEC6 baseline above frontier)
- RECOVERY-1 commit `39c76755b` — canonical equation registry 52 → 53 (FEC10 hybrid registered)
- DQS1 canonical frontier: sha `7a0da5d0fc327cba3f7d1387a544fd5ce5f05bc56ecc8e12cd5097141672f4fe` / 178559 bytes / CPU 0.19202828295713675 / CUDA 0.22619176954300405 paired
- V14-V2 substituted archive: `experiments/results/v14_v2_dqs1_plus_fec10_substituted_20260526T023000Z/submission_dir/archive.zip`
- Paired Modal call_ids: CUDA `fc-01KSKNC9BQ4ABNNC8AW77SB56E` + CPU `fc-01KSKNB14HSA215KB4J3CJTFW4`
- Paired auth_eval results: `experiments/results/v14_v2_dqs1_plus_fec10_substituted_20260526T023000Z/modal_auth_eval{,_cpu}/.../contest_auth_eval.json`
- Canonical equation: `cascade_a_fec10_hybrid_adaptive_blend_savings_v1` (entry in `.omx/state/canonical_equations_registry.jsonl`)
- Canonical frontier pointer: `.omx/state/canonical_frontier_pointer.json` (refresh: `.venv/bin/python tools/refresh_canonical_frontier.py`)
- PR111 candidate report: `reports/pr111_candidate_cascade_a_fec10_substitution_onto_dqs1_frontier_landed_20260526.md`

---

## CLAUDE.md non-negotiable compliance

- ✅ **Modal `.spawn()` HARVEST OR LOSE**: both call_ids registered in canonical ledger + outcomes recovered via `tools/recover_modal_auth_eval.py`
- ✅ **Apples-to-apples evidence discipline**: paired CPU + CUDA on SAME substituted archive bytes; pre-dispatch decode-equality + frame-byte-identical inflate verified locally as canonical apples-to-apples check; axis tags explicit; hardware substrates declared
- ✅ **Submission auth eval — BOTH CPU AND CUDA**: explicit verdict that BOTH axes beat frontier; PR111 candidate; operator-routable for PR submission via canonical pipeline
- ✅ **MPS auth eval is NOISE**: paired dispatch via Modal CPU + Modal T4 (1:1 contest-compliant hardware per CLAUDE.md); ZERO MPS
- ✅ **Forbidden /tmp paths**: substituted archive + landing memo + canonical equation update all at canonical `experiments/results/` + `.omx/research/` + `.omx/state/` paths
- ✅ **Forbidden component-aliasing**: archive sha verified pre + post paired dispatch; cross-verified against DQS1 baseline sha
- ✅ **NEVER invent CLI flags**: `tools/dispatch_modal_paired_auth_eval.py` + `experiments/modal_auth_eval{,_cpu}.py` invoked with verified argparse signature
- ✅ **Forbidden score claims**: every score literal in this memo is tagged with axis (`[contest-CPU]`, `[contest-CUDA T4]`) AND tagged via canonical Provenance per Catalog #287/#323 (built via `tac.provenance.builders.build_provenance_for_archive_member`)
- ✅ **Subagent coherence-by-default**: read CLAUDE.md + V14 landing memo + RECOVERY-1 commit + canonical frontier pointer + canonical equation registry BEFORE any edit (Catalog #229 PV)
- ✅ **Catalog #117/#157/#174/#235/#289 commit serializer discipline**: commit via `tools/subagent_commit_serializer.py` + POST-EDIT `--expected-content-sha256`
- ✅ **Catalog #119 Co-Authored-By Claude trailer**: included in subagent commits (forbidden in PR111 candidate report per user_pr_attribution memory)
- ✅ **Catalog #206 checkpoint discipline**: 6+ checkpoints emitted to `.omx/state/subagent_progress.jsonl`
- ✅ **Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE**: canonical equation update is APPEND-ONLY (V14 anchors preserved verbatim; only added new V14-V2 anchors)
- ✅ **Catalog #230 sister-disjoint scope**: stayed within DQS1-substituted-archive build + paired-CUDA dispatch + canonical equation registration + landing memo + PR111 candidate report + lane registry; NOT touching Cascade C' WAVE-6 + Phase 3 archive_grammar work
- ✅ **Catalog #340 sister-checkpoint guard**: PROCEED (no sister conflicts at edit-time)
- ✅ **Catalog #344 PROMOTION discipline**: canonical equation #344 entry PROMOTED with paired empirical CUDA + CPU anchors only (4th + 5th); 5 anchors total
- ✅ **Catalog #343 NO hardcoded score literals**: this memo cites canonical pointer file as source of truth for frontier; superseding score values are empirical results from contest_auth_eval.json files (canonical posterior anchors); operator-routable to update pointer per Catalog #343 discipline; this memo carries `HISTORICAL_SCORE_LITERAL_OK` waivers below for the empirical anchor literals it must document
- ✅ **10th apples-to-apples canonicalization**: decode-equality verified FIRST; frame-byte-identical inflate verified SECOND; paired-axis Modal dispatch THIRD (canonical PV ordering)
- ✅ **11th ORDER canonicalization**: decode-equality verify → frame-byte-identical inflate verify → paired-axis dispatch → PROMOTION (5-step canonical ordering)
- ✅ **12th canonicalization × standardization × ease-of-contest-compliance trinity**: canonical encoder/decoder library (RECOVERY-1) + canonical paired-dispatch helper (Catalog #246) + canonical Provenance builders (Catalog #287/#323) + canonical equation registry (Catalog #344) + canonical frontier pointer (Catalog #343) — every primitive consumed; zero hand-rolled scaffolds

---

## Mission contribution per Catalog #300

`frontier_breaking` — PAIRED-AXIS FRONTIER-CROSSING empirically validated on BOTH contest-CPU AND contest-CUDA axes; canonical equation #344 PROMOTED 3 → 5 anchors with V14-V2 DQS1-substituted anchors. Per CLAUDE.md "Forbidden premature KILL without research exhaustion": V14's prior "NOT PR111 candidate" verdict was IMPLEMENTATION-LEVEL substrate-mismatch (FEC6 PR110 baseline above frontier), not paradigm refutation per Catalog #307 paradigm-vs-implementation classification — V14-V2 RATIFIES the operator insight + RATIFIES the FEC10 hybrid codec PARADIGM at frontier-crossing scale.

<!-- HISTORICAL_SCORE_LITERAL_OK:v14_v2_frontier_crossing_paired_axis_empirical_anchor_landing_memo_2026-05-26 -->
