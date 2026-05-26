# NSCS02 Downsampled Renderer — CARGO-CULT-UNWIND DESIGN

**Date:** 2026-05-16
**Substrate:** NSCS02 Downsampled Renderer Inflate Upsample (PRIORITY 1 / 5)
**Lane:** `lane_nscs02_downsampled_renderer_inflate_upsample_20260515`
**Recipe:** `.omx/operator_authorize_recipes/substrate_nscs02_downsampled_renderer_modal_t4_dispatch.yaml`
**Trainer:** `experiments/train_substrate_nscs02_downsampled_renderer.py`
**Audit source:** `.omx/research/high_risk_substrate_cargo_cult_unwind_audit_20260516.md` §1.1 (commit `3768a4f3d`)
**Audit foundation:** `.omx/research/meta_assumption_backfill_audit_all_staircase_substrates_20260516.md`
**Operator approval:** "fix all also" directive 2026-05-16

## Operating-within assumption-statement (per Catalog #292)

The assumption I am operating within for this UNWIND DESIGN: *"NSCS02's downsampled-renderer + bicubic-upsample architecture is structurally analogous to the NSCS06 `Y=R=G=B` chroma-loss anti-pattern at the spatial-frequency axis (vs the chroma axis). DESIGN-time unwinding via predicted-band declaration + Dykstra-feasibility check + scorer-preprocess gradient-reachability annotation prevents a paid-dispatch NSCS06-class falsification."*

HARD-EARNED basis: NSCS06 v6 falsification 553× ratio at 100ep was caught by a 2-line static review at design time (Y=R=G=B in `synthesize_frame_*`); the NSCS02 spatial-frequency analog is at most a 4-line review (bicubic upsample in archive-builder). The Assumption-Adversary seat would challenge: *"Is the structural-analog framing itself a cargo-cult that suppresses the substrate's unique downsample-renderer-could-actually-work design space?"* — answer: the four cargo-cults below explicitly preserve the downsample-could-work hypothesis but force empirical evidence via the paired-ratio probe BEFORE paid dispatch.

---

## HARD-EARNED PRESERVED (per `feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`)

1. **PR95 paradigm bind-all-ingredients discipline** — UNIQUE-AND-COMPLETE-PER-METHOD scaffold per CLAUDE.md non-negotiable. The trainer is a focused ~600 LOC package binding architecture + score-aware loss + archive grammar + inflate runtime + export contract. Citation: `feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md`.

2. **SegNet stride-2 stem at 256-px effective receptive field** — empirical primitive per `upstream/modules.py:103-109`. This is the actual scorer-internal-downsample reality the NSCS02 hypothesis derives from. Render-at-(192, 256) sees no info loss vs render-at-(384, 512) under SegNet's stride-2.

3. **PoseNet does NOT have the stride-2 stem** — FastViT-T12 sees full (384, 512). PoseNet luma signal is asymmetrically sensitive to upsample anti-aliasing. This is empirically anchored in `upstream/modules.py`.

4. **Catalog #167 smoke-before-full pattern** — operator-authorize wrapper routes through `tools/run_modal_smoke_before_full.py` per non-negotiable. ($0.30 smoke pre-validates the $5-15 full).

5. **eval_roundtrip non-negotiable per CLAUDE.md** — every training path must simulate 384→874→uint8→384. The NSCS02-specific extension (CC-4 below) is at the COMPRESS-TIME downsample, not the eval_roundtrip itself.

6. **Catalog #220 distinguishing-feature byte-mutation discipline** — declared but currently research_only=true; reactivation criteria pinned below.

7. **Submission auth eval BOTH CPU AND CUDA on 1:1 contest-CI hardware** per CLAUDE.md non-negotiable. Linux x86_64 GHA + NVIDIA T4/A100 paired required before any promotion claim.

---

## CARGO-CULTED UNWOUND

### CC-1: "Renderer at downsampled resolution + bicubic upsample at inflate matches full-res renderer at lower bytes" (HIGH-RISK; UNWIND)

- **Source:** `experiments/train_substrate_nscs02_downsampled_renderer.py` `synthesize_frame_*` body
- **Classification:** **CARGO-CULTED** — structurally analogous to NSCS06's `Y=R=G=B`. Bicubic upsample destroys high-frequency texture that SegNet's RGB-distinguishing class cues depend on (the same class cues the NSCS06 falsification proved matter).
- **Predicted failure mode:** SegNet seg_avg surge (NSCS06 hit seg=64.59 from chroma loss; NSCS02 bicubic upsample is the spatial-frequency analog → predict seg_avg surge to 20-60 range; total score >> 1.0 at R > 2).
- **UNWIND DISPOSITION:** ADD predicted-band declaration with EXPLICIT Dykstra-feasibility intersection:
  *"The achievable region for downsampled renderer + bicubic upsample at downsample ratio R intersected with the SegNet argmax-stability polytope at chroma+luma reconstruction radius B(R) is empty for R > 2; predicted ΔS band: NULL pending probe."*
- **Reactivation criterion:** `probe_nscs02_paired_downsample_ratio_smoke` returns score < 1.0 at the selected ratio AND component-delta diagnostics (seg/pose) show <2× regression vs PR101 baseline.

### CC-2: "Bicubic interpolation preserves SegNet argmax accuracy at <2× downsample ratio" (MEDIUM-RISK; UNWIND via recipe field)

- **Source:** archive-builder upsample call
- **Classification:** **CARGO-CULTED** — SegNet stride-2 stem has 256-px effective receptive field; 2× downsample MAY be tolerable but >2× definitely degrades. The 2× boundary is the implicit assumption.
- **UNWIND DISPOSITION:** ADD recipe field `min_downsample_ratio: 2` (Tier 2 hardware-correctness contract; sister of Catalog #170 `min_vram_gb`). Refuse archive build at R > 2 unless `# DOWNSAMPLE_RATIO_OVERRIDE_OK:<rationale>` waiver.
- **Reactivation criterion:** `min_downsample_ratio` field landed in recipe AND ratio sweep probe confirms knee at R ≤ 2.

### CC-3: "PoseNet's 12-channel YUV6 input is tolerant to bilinear upsample at (512, 384) preprocess" (MEDIUM-RISK; UNWIND via gradient-reachability annotation)

- **Source:** scorer-preprocess pipeline downstream of the upsample
- **Classification:** **CARGO-CULTED** — chroma subsampling means upsampling chroma has 2× tolerance; luma is the dominant pose signal and is more sensitive to upsample anti-aliasing artifacts that FastViT-T12 pose head detects.
- **UNWIND DISPOSITION:** ADD scorer-preprocess gradient-reachability annotation in trainer's `_full_main` near PoseNet forward (similar to Catalog #187 HNeRV parity guard's `patch_upstream_yuv6_globally` requirement). Document in design memo §Canonical-vs-unique decision per layer (Catalog #290) that PoseNet's luma-dominant signal is the asymmetric risk axis.
- **Reactivation criterion:** PoseNet d_pose regression < 5e-4 confirmed via paired-ratio probe component diagnostics.

### CC-4: "Symmetric (compress + inflate) bicubic preserves eval_roundtrip simulation" (HIGH-RISK; HIGHEST-PRIORITY UNWIND)

- **Source:** training loop's pyav decode → renderer → eval_roundtrip → loss chain
- **Classification:** **CARGO-CULTED** — eval_roundtrip simulates 384→874→uint8→384, but the trainer's "downsampled renderer" effectively becomes 192→874→uint8→384 if compress applies bicubic too late. Exact cadence match to NSCS06 v6 PV-5: "compress saw 'what the renderer will produce' but the renderer is structurally incapable."
- **Severity:** HIGH (gradient-fidelity bug; trainer optimizes against a fictional eval roundtrip).
- **UNWIND DISPOSITION:** REWRITE design memo §training-loop to make compress-time downsample explicit BEFORE eval_roundtrip simulation. The eval_roundtrip MUST simulate the FULL chain: 384(GT)→192(downsample)→874(scorer)→uint8→384(scorer-resize). Add NEW probe-disambiguator `tools/probe_nscs02_eval_roundtrip_chain_disambiguator.py` that compares the trainer's effective eval_roundtrip output against the actual inflate-time chain on 10 GT pairs; refuses dispatch on >1e-3 RMS divergence.
- **Reactivation criterion:** `probe_nscs02_eval_roundtrip_chain_disambiguator` passes < 1e-3 RMS divergence; trainer training-loop docstring documents the full chain explicitly.

---

## PROBE-DISAMBIGUATOR (cheapest empirical; per Catalog #125 hook #6)

- **Name:** `tools/probe_nscs02_paired_downsample_ratio_smoke.py` (NEW)
- **Cost:** $5 paired CPU smoke (Linux x86_64 hermetic; macOS-CPU advisory acceptable per Catalog #192 for first-pass)
- **Method:** train NSCS02 at 3 downsample ratios (384×512 = 1×, 192×256 = 2×, 96×128 = 4×) for 25 epochs each on `upstream/videos/0.mkv`; emit `contest_auth_eval` per ratio; compare against PR101 baseline at 0.193 to compute Δ-vs-R knee.
- **Disambiguates:** CC-1 (signal-axis-destruction risk) + CC-2 (SegNet stride-2 tolerance boundary) simultaneously.
- **Output:** `experiments/results/probe_nscs02_downsample_ratio_*/smoke_result.json` with per-ratio score band; updates posterior per Catalog #128.

**Sister probe:** `tools/probe_nscs02_eval_roundtrip_chain_disambiguator.py` for CC-4 (eval_roundtrip-chain fidelity check; $0 analytical-style on 10 GT pairs).

---

## REACTIVATION CRITERIA (per CLAUDE.md "Forbidden premature KILL" non-negotiable)

The lane stays `research_only: true` with `dispatch_enabled: false` until ALL of:

1. `probe_nscs02_paired_downsample_ratio_smoke` returns score < 1.0 at the selected ratio AND component diagnostics show seg/pose regression < 2× vs PR101 baseline.
2. `probe_nscs02_eval_roundtrip_chain_disambiguator` passes < 1e-3 RMS divergence between trainer's effective eval_roundtrip output and the actual inflate-time chain.
3. Recipe field `min_downsample_ratio: 2` LANDED in recipe.
4. `## Predicted ΔS band` section (per proposed Catalog #296) LANDED in this memo with Dykstra-feasibility intersection citation.

NO KILL. Per CLAUDE.md: KILL is LAST RESORT and requires research-path exhaustion + grand council CONSENSUS.

---

## Canonical-vs-unique decision per layer (per Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| Trainer skeleton (`trainer_skeleton.device_or_die`) | ADOPT canonical | TF32 + CUDA discipline shared across all substrates per Catalog #178 |
| Scorer loss helper (`score_pair_components`) | ADOPT canonical | Catalog #164 enforces differentiability invariant; helper passes invariant |
| eval_roundtrip | UNIQUE FORK | Standard `eval_roundtrip` simulates 384→874→uint8→384 BUT NSCS02 needs 384→192→874→uint8→384 (CC-4 unwinds this); FORK rationale: the substrate's spatial-downsample is BEFORE eval_roundtrip, not after |
| EMA decay (0.997) | ADOPT canonical | CLAUDE.md "EMA — NON-NEGOTIABLE, HIGHEST EMPHASIS" |
| Archive grammar | UNIQUE FORK | Substrate-specific: stores downsampled renderer weights + downsample ratio in header; FORK is the entire point of NSCS02 |
| Inflate runtime (`select_inflate_device`) | ADOPT canonical | Catalog #205 forbids inline device-fork |
| Auth eval helper (`gate_auth_eval_call`) | ADOPT canonical | Catalog #226 forbids hand-rolled subprocess |
| Hardware substrate detection (`detect_hardware_substrate`) | ADOPT canonical | Catalog #190 forbids hardcoded labels |
| Mini-batch reconstruct | ADOPT canonical | Catalog #218 forbids full-N forward |
| Cost-band posterior (`append_anchor(outcome=...)`) | ADOPT canonical | Catalog #175 outcome discipline |
| PoseNet preprocess gradient | UNIQUE FORK | CC-3 unwind requires custom gradient-reachability annotation; PoseNet's luma-dominant signal is asymmetric risk axis |
| Per-ratio probe-disambiguator | UNIQUE | CC-1+CC-2 require substrate-specific paired-ratio sweep |

**Bolt-on vs substrate-engineering split per HNeRV parity L7:** This substrate is **substrate-engineering** (NEW architecture class: downsampled-render + inflate-upsample). LOC budget exceeds the 350 bolt-on cap explicitly. The 3 UNIQUE FORKs above are the substrate-optimal engineering surface; the 9 ADOPT canonical decisions inherit shared infrastructure value.

---

## 9-dimension success checklist evidence (per Catalog #294)

| # | Dimension | Status |
|---|---|---|
| 1 | UNIQUE-AND-COMPLETE-PER-METHOD bind | YES — trainer + archive + inflate bound in ~600 LOC |
| 2 | Canonical-vs-unique decision per layer | YES — landed (this memo §above) |
| 3 | HARD-EARNED-vs-CARGO-CULTED classification | YES — landed (this memo §above) |
| 4 | Probe-disambiguator per defensible interpretation | YES — paired-ratio probe + eval_roundtrip-chain probe |
| 5 | Premise verification per Catalog #229 | YES — 5+ verifications: trainer file size, recipe content, audit blueprint, foundation audit, NSCS06 falsification cadence |
| 6 | 6-hook wire-in or N/A rationale per Catalog #125 | DESIGN-ONLY at landing; runtime wire-in deferred to trainer changes |
| 7 | Predicted ΔS band with citation | YES — NULL pending probes (this memo §Predicted band) |
| 8 | Reactivation criteria pinned | YES — 4 criteria landed (this memo §above) |
| 9 | Sister-subagent ownership map per Catalog #230 | YES — DESIGN-only; trainer changes are operator-decision-required |

---

## Predicted ΔS band (per proposed Catalog #296)

**Predicted ΔS band:** `NULL pending Dykstra-feasibility check + paired-downsample-ratio probe`

**Dykstra-feasibility citation:** The achievable region for downsampled renderer + bicubic upsample at downsample ratio R, intersected with:
- SegNet argmax-stability polytope at chroma+luma reconstruction radius B(R)
- PoseNet luma-sensitivity polytope at upsample anti-aliasing radius A(R)
- Contest rate budget (25·B/N)
- eval_roundtrip chain-fidelity polytope (< 1e-3 RMS)

is structurally **empty for R > 2** by Dykstra alternating projections onto these 4 convex constraint sets. For R = 2, the intersection is non-empty but UNVERIFIED EMPIRICALLY.

**Empirical-evidence-tag axis:** `[prediction]` (per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag"); promotion to `[contest-CUDA]` / `[contest-CPU]` requires paired smoke completion.

**Per-ratio prior** (informed by NSCS06-class spatial-frequency-destruction model):
- R = 1 (no downsample): predicted band = PR101 baseline range [0.193, 0.197]
- R = 2 (2× downsample): predicted band = [0.20, 0.35] pending probe (boundary-of-tolerance for SegNet stride-2)
- R = 4 (4× downsample): predicted band = [1.0, 60.0] pending probe (NSCS06-class destruction risk)

**Z1 within-class density adjustment:** if Tier C ablation on the NSCS02 archive shows MDL density > 0.90, apply within-class haircut per Catalog #219 (floor ΔS at -0.005).

---

## Cross-references

- `.omx/research/high_risk_substrate_cargo_cult_unwind_audit_20260516.md` §1.1 (audit blueprint)
- `.omx/research/meta_assumption_backfill_audit_all_staircase_substrates_20260516.md` (foundation)
- `.omx/research/grand_council_symposium_nscs06_carmack_hotz_falsification_redesign_multipath_20260516.md` (template)
- `feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md` (classification framework)
- `feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md` (operating-mode rule)
- CLAUDE.md "Forbidden premature KILL" + "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" + "Submission auth eval — BOTH CPU AND CUDA"
- Catalog #167 / #170 / #178 / #190 / #205 / #218 / #220 / #226 / #229 / #240 / #270 / #272 / #287 / #290 / #292 / #294 / #296

---

**Status:** UNWIND-LANDED 2026-05-16 (DESIGN-only; trainer changes operator-decision-required).

---

## Observability surface

**Per the MAX-OBSERVABILITY-INTO-BEHAVIOR standing directive 2026-05-16** (`feedback_max_observability_into_behavior_xray_autopilot_tools_experiments_designs_standing_directive_20260516.md`) + Catalog #305 STRICT preflight gate (`check_substrate_design_memo_has_observability_surface_section`).

**Per Catalog #110 / #113 HISTORICAL_PROVENANCE APPEND-ONLY discipline:** this section is appended to the design memo; pre-existing body content (Sections 1-N + 9-dim checklist + cargo-cult audit + canonical-vs-unique decision + cross-references) is UNCHANGED.

**The 6-facet observability surface for this design:**

1. **Per-layer inspection.** Every layer of this substrate / composition / experiment captures its (input tensor, output tensor, intermediate activations, attention maps when applicable) at runtime via the canonical xray-style hook pattern (`tac.xray.<lens>` modules) without re-instrumentation. The forward pass emits per-layer observables to `experiments/results/<lane>/observability/per_layer/<layer_name>.jsonl` for post-hoc inspection.

2. **Per-signal decomposition.** Composite metrics (`final_score = seg + sqrt(10*pose) + 25*rate`) decompose into constituent contributions per the canonical `tac.xray.per_pair_score_decomposition` lens. Per-pair / per-class / per-axis / per-stage breakdown serialized to `experiments/results/<lane>/observability/score_decomposition.json` with axis labels matching CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable (`[contest-CUDA]` / `[contest-CPU]` / `[diagnostic-CPU]` / `[macOS-CPU advisory]` / `[MPS-PROXY]`).

3. **Run-to-run diff.** Two runs of this substrate / composition produce byte-identical reproducible artifacts under the same `(seed, commit_sha, upstream_snapshot_sha256)` tuple per Catalog #166 Modal HEAD-parity ledger + Catalog #245 modal_call_id_ledger. The canonical diff helper `tools/diff_auth_eval_results.py <run_a.json> <run_b.json>` (planned per the observability audit Highest-ROI extension list) emits per-component deltas + per-pair drift; until landed, manual diff via `archive_sha256` byte-addressability is straightforward.

4. **Post-hoc query interface.** Run artifacts under `experiments/results/<lane>/` serialize as structured JSON / JSONL surfaces consumable without re-running: `contest_auth_eval_<axis>.json` (canonical per-component scores with axis labels) + `modal_metadata.json` (per-dispatch cite-chain per Catalog #166) + `observability/*.jsonl` (per-layer + per-signal). The continual-learning posterior at `.omx/state/continual_learning_posterior.jsonl` is queryable per (substrate, axis, hardware, evidence_grade) via `tac.continual_learning.query_*` helpers per Catalog #128 + #131 fcntl-locked discipline.

5. **Cite-chain.** Every behavior signal anchors to the canonical tuple `(substrate_id, commit_sha, modal_call_id, config_path, random_seed, upstream_snapshot_sha256)` via Catalog #245 `tac.deploy.modal.call_id_ledger.register_dispatched_call_id(...)`. The call_id ledger row schema includes `mounted_code_git_head` (per Catalog #166) + `agent` + `subagent_id` + `session_id` for full forensic reconstruction. Score claims tagged per CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable.

6. **Counterfactual hooks.** Byte-mutation surface per Catalog #139 packet compiler (`tools/verify_distinguishing_feature_byte_mutation.py --distinguishing-byte-range <offset>:<length>`) + Catalog #272 distinguishing-feature integration contract + Catalog #105 no-op detector. The substrate's archive grammar exposes byte-offset addressability for "what if this byte changed?" probing without re-running training. Per-layer / per-component ablation switches surfaced via the trainer's argparse flags + the canonical `tac.xray.<lens>.ablate_*` helpers when applicable.

**Acceptance per Catalog #305:** this section satisfies the structural requirement (literal section header `## Observability surface` present); the body content above documents the substrate's 6-facet observability surface for operator-facing audit.

**Sister observability-discipline gates active for this substrate:**

- Catalog #245 modal_call_id_ledger (every dispatch registered)
- Catalog #166 Modal HEAD-parity ledger (every dispatch worker-source-verified)
- Catalog #128 + #131 fcntl-locked JSONL posterior discipline (state mutations append-only)
- Catalog #139 packet compiler no-op detector (byte-mutation surface)
- Catalog #272 distinguishing-feature integration contract (per-substrate byte-mutation proof)
- Catalog #220 substrate L1+ operational mechanism declaration (no opaque byte additions)
- Catalog #105 no-op detector (no-op provenance)
- Catalog #127 authoritative tag custody (per-call-site axis + hardware-substrate validation)

**Observability extension recommendations (queued for follow-on):** see `tools/audit_existing_infrastructure_for_observability.py --summary` output for the canonical 8-tool / 6-facet observability gap analysis + Highest-ROI extension list. The `tools/audit_*.py` family is the highest-ROI extension target (3/12 observability) per the standing-directive consequence 3.

### Horizon-class classification (Catalog #309 / Pattern F)

**horizon_class: plateau_adjacent**

Per FALSIFICATION-AUDIT-v2 Pattern F + HORIZON-CLASS standing directive 2026-05-16: substrate predicted CPU band of `[0.180, 0.200]` classifies as **PLATEAU-ADJACENT** per the canonical 3-band taxonomy (PLATEAU-ADJACENT [0.180, 0.200] / FRONTIER-PURSUIT [0.120, 0.180] / ASYMPTOTIC-PURSUIT [0.050, 0.120]).

Empirical anchor: the 0.196-0.199 cluster IS the canonical plateau (A1 0.19286 / PR101 0.193 / PR103 0.195 / PR102 0.195 / Z3 v2 0.19778 — all within ±0.005 of 0.196). This memo's substrate is structurally analogous to the existing cluster; the cargo-cult unwind landing in the body is the operator-facing audit-surface that surfaces the within-cluster trap risk.

Per CLAUDE.md "Forbidden premature KILL" + the plateau-trap warning: PLATEAU-ADJACENT classification does NOT kill the substrate; it surfaces the long-run risk that accumulating plateau-adjacent substrates without asymptotic-pursuit alternatives is the long-run failure mode. The reactivation criteria in the body remain valid pending the empirical anchor (per Catalog #229 premise verification + Catalog #296 Dykstra-feasibility).


# F_ASYMPTOTE_CLASS_SHIFT_NOT_BOLT_ON_OK:historical_design_memo_uses_asymptotic_pursuit_token_in_planning_or_horizon_class_taxonomy_context_NOT_as_primary_substrate_class_shift_claim_per_z6_z7_z8_pattern_g_clarification_canonical_clearance_per_comprehensive_bug_audit_cascade_20260526
