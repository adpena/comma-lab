# Codex routing directive TOP-1: A1-SPECIALIZED per-pattern VQ-VAE inverter prototype
# Date: 2026-05-18
# Authority: SYSTEMATIC RECLAIMABILITY RE-EXAMINATION verdict PROCEED_WITH_REVISIONS (commit `4480d9b14`; T2 sextet + engineering quartet + 3 grand-council seats; 13 attendees) + A1 binary distillation council PROCEED_WITH_REVISIONS (commit `0701c323b`; T2 sextet + Carmack/Hotz/Quantizr/van den Oord engineering quartet; 5 binding revisions)
# Budget envelope: $1-3 paid GPU spend; OPERATOR AUTHORIZATION REQUIRED before execution
# Predicted [prediction] ΔS band: [-0.012, -0.003]; predicted compressed binary 5-20 KB
# Lane class: ASYMPTOTIC-PURSUIT reclamation; first vector in TOP-3 EV ordering

## CANONICAL POINTERS

1. `/Users/adpena/Projects/pact/CLAUDE.md` (FULL; especially "Forbidden premature KILL without research exhaustion" + "HNeRV / leaderboard-implementation parity discipline" L4 ≤200 LOC inflate.py + "Contest vs production target modes" `contest_one_video_replay` + Catalog #270/#325/#313/#167/#199 dispatch discipline cluster)
2. `/Users/adpena/Projects/pact/AGENTS.md`
3. `.omx/research/systematic_reclaimability_re_examination_via_binary_distillation_framework_20260518.md` (commit `4480d9b14`; TOP-3 source-of-truth)
4. `.omx/research/a1_binary_distillation_design_memo_zig_sparseness_ablation_plus_canonical_techniques_20260518.md` (commit `0701c323b`; canonical framework + 5 binding revisions)
5. `.omx/research/rate_attack_legal_receiver_path_audit_codex_f1_finding_relay_20260518.md` (commit `35b06f9ec`; A1-SPECIALIZED is RECLAIMABLE_VIA_PACKET_COMPILER per Codex Section 0)
6. `.omx/state/probe_outcomes.jsonl` (consult per Catalog #313 BEFORE firing)
7. `tools/operator_authorize.py` (canonical entry point; will gate via Catalog #270 + #325 symposium + #313 predecessor check + #167 smoke-before-full)

## STRATEGIC FRAMING

**A1-CANONICAL** was originally classified `STRICT_SCORER_RULE_VIOLATION` per the 43-vector audit (naive framing assumed full-PoseNet weights at inflate). **A1-SPECIALIZED** is the same scorer-feature-space encoding paradigm but distilled to a per-pattern specialized binary: VQ-VAE codebook (K=256) + FP4 quantization + 50% sparseness + Brotli. The specialized binary "obviously cannot be re-used as PoseNet for arbitrary input" (Yousfi PR #35 author interpretation cited in `0701c323b`) — it's a `contest_one_video_replay` mode artifact per CLAUDE.md "Contest vs production target modes" non-negotiable.

Predicted realistic compressed size: 5-20 KB (theoretical floor ~1.7-4 KB; aggressive variants could sub-1 KB per V3 weight-derived path). Predicted contest-CUDA / contest-CPU ΔS: [-0.012, -0.003] (cite `4480d9b14` per-vector matrix row).

This is the cheapest + highest-EV TOP-3 path. Empirical anchor unlocks the F4 + F5 sister paths (TOP-2 + TOP-3) which use the same framework on different feature spaces.

## 6 BINDING REVISIONS (from `0701c323b` T2 council; carry into Codex execution)

1. **OP-A1-BIN-1**: PROCEED on per-pattern distilled path; NOT generic Hinton student
2. **OP-A1-BIN-2**: TOP-1 composition = V2 (VQ-VAE K=256 + FP4 + 50% sparse + Brotli) → 5-10 KB target
3. **OP-A1-BIN-3**: 4 reactivation conditions pinned per Catalog #325 (cite memo for full list)
4. **OP-A1-BIN-4**: Reject pure-Zig/Rust binary path (Python in pinned env; ELF/Mach-O overhead > Python's marginal cost)
5. **OP-A1-BIN-5**: 1 mandatory empirical anchor before Phase 2 ($1-3 prototype) — THIS DIRECTIVE EXECUTES THAT ANCHOR

## 6-PHASE EXECUTION

### Phase 1: Predecessor-probe consultation (per Catalog #313)

```python
from tac.probe_outcomes_ledger import latest_blocking_outcome_by_substrate

verdict = latest_blocking_outcome_by_substrate(substrate_id="a1_specialized_per_pattern_vq_vae_inverter")
if verdict and verdict.status == "blocking":
    raise SystemExit(f"Catalog #313: blocking predecessor probe outcome — {verdict.rationale_path}; cite override OR ratify")
```

No predecessor at landing time; proceed to Phase 2.

### Phase 2: Build canonical prototype helper

Build `tools/build_a1_per_pattern_vq_vae_inverter_prototype.py` (~300-500 LOC):
- Reads A1 archive bytes + canonical scorer feature extractor (PoseNet `summary(512)` slot per upstream/modules.py — verify via Catalog #229 premise check before encoder design)
- Per-pattern VQ-VAE: K=256 codebook quantization on the substrate's pattern manifold (NOT the full feature space; per-pattern = per-(video-segment, scene-cluster) specialization per `0701c323b` framework)
- FP4 quantization on codebook entries (via canonical `tac.quantization.FakeQuantFP4`)
- 50% magnitude-pruning sparseness mask
- Brotli compression on the resulting blob
- Output: a `<5-20 KB` `.bin` artifact suitable for inflation alongside Lane A archive

### Phase 3: Empirical bit-spend smoke (T4 CPU or cheap GPU; $1-3 envelope)

Per Catalog #167 smoke-before-full pattern:
- Smoke target: 100 patterns × ~10 seconds each on T4 (Modal T4 ≈ $0.30 / 100s + overhead; envelope ≈ $1-3)
- Measure actual compressed bytes vs predicted 5-20 KB range
- Measure proxy distortion (Hinton-distilled SegNet surrogate via Catalog #523 if available; otherwise skip proxy)
- Fail-fast threshold: if compressed bytes > 25 KB OR proxy distortion >2× predicted, ABORT before full empirical run

Per Catalog #270 canonical dispatch protocol: trainer + recipe MUST declare Tier 1+2+3 fields before paid dispatch fires. Smoke recipe template at `.omx/operator_authorize_recipes/substrate_a1_specialized_per_pattern_vq_vae_inverter_modal_t4_smoke.yaml` (Codex to create per canonical template + sister Catalog #244 NVML env block + #240 recipe-vs-trainer-state consistency).

### Phase 4: Measurement on Lane A + Lane 12 + apogee_v2 archives

Per the 43-vector audit's anchor pattern + A1 council Revision #4 (REJECT pure-Zig path; use the canonical compressed-binary approach):
- Lane A archive (canonical target)
- Lane 12 sister (NeRV mask codec; sister substrate)
- apogee_v2 archive (NSCS06 v6 cargo-cult-unwind canonical reference; commit `90bca47ff` post-fix state)

Measure per-archive: compressed bytes + contest-CUDA + contest-CPU score deltas vs baseline. Use canonical `experiments/contest_auth_eval.py` via `tac.substrates._shared.smoke_auth_eval_gate.gate_auth_eval_call(...)` per Catalog #226.

### Phase 5: Wrap output in canonical Provenance per Catalog #323

**VERIFIED canonical API** (per `dir(tac.provenance)` 2026-05-18 inspection; see commit `ecaa1c471` 14th META-audit instance):

```python
from tac.provenance import (
    build_provenance_for_archive_member,
    provenance_to_dict,
    ProvenanceEvidenceGrade,
)

# After empirical anchor lands:
prov = build_provenance_for_archive_member(
    archive_zip_path="experiments/results/a1_specialized_inverter_<utc>/archive.zip",
    member_name="inverter.bin",
    measurement_axis="contest_cuda",  # or contest_cpu per axis being reported
    hardware_substrate="linux_x86_64_t4",  # or actual GPU detected at runtime
    evidence_grade=ProvenanceEvidenceGrade.PROMOTABLE_EXACT_CONTEST_CUDA,
)

result_row = {
    "substrate": "a1_specialized_per_pattern_vq_vae_inverter",
    "compressed_bytes": <int>,
    "score": <float>,
    "score_axis": "contest_cuda",
    "hardware_substrate": "linux_x86_64_t4",
    "evidence_grade": "contest-CUDA",
    "archive_sha256": "<sha>",
    "provenance": provenance_to_dict(prov),
}
```

Bonus structural protection: `build_provenance_for_archive_member` REFUSES construction when archive path does not exist on disk (fail-closed at builder surface).

Pre-empirical predicted rows MUST use `build_provenance_for_predicted(model_id, inputs_sha256, ...)` with `measurement_axis='[predicted]'` per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" non-negotiable.

### Phase 6: Register probe outcome per Catalog #313

```python
from tac.probe_outcomes_ledger import register_probe_outcome

register_probe_outcome(
    probe_id="a1_specialized_per_pattern_vq_vae_inverter_prototype_20260518",
    verdict="<PROMOTE|PROCEED|DEFER|INDEPENDENT|KILL>",  # per empirical result
    status="adjudicated",
    rationale_path="experiments/results/a1_specialized_inverter_<utc>/report.json",
    expires_at_utc="<+30 days>",
    agent="codex",
    notes="A1-SPECIALIZED V2 (VQ-VAE K=256 + FP4 + 50% sparse + Brotli). Compressed bytes=<N>. Score delta=<dx>. Per `4480d9b14` SYSTEMATIC RECLAIMABILITY TOP-1 + `0701c323b` A1 binary distillation council 5-binding-revisions."
)
```

## DISCIPLINE

- Catalog #229 premise verification BEFORE Phase 2 (read upstream/modules.py:26 + :84 + 0701c323b VQ-VAE framework spec; verify before building)
- Catalog #287 evidence tags on every score claim (`[contest-CUDA]` / `[contest-CPU]` / `[prediction]`)
- Catalog #117/#157/#174 commit serializer with POST-EDIT sha256
- Catalog #186 NO new catalog # needed
- Catalog #206 checkpoint discipline (Codex /goal LOOP provides)
- Catalog #270 dispatch optimization protocol (Tier 1+2+3 declarations in recipe before paid dispatch)
- Catalog #325 per-substrate symposium: SATISFIED via `4480d9b14` PROCEED_WITH_REVISIONS T2 council (within 14-day window)
- Catalog #313 probe-outcomes ledger consultation + registration
- Catalog #167 smoke-before-full
- Catalog #199 paired-env operator bypass discipline if non-interactive
- Catalog #244 canonical NVML env block in driver script
- Catalog #226 canonical auth-eval helper routing
- Catalog #205 canonical inflate device-fork (`PACT_INFLATE_DEVICE`)
- Catalog #316 frontier scan post-empirical: register strengthened verdict
- Catalog #323 canonical Provenance contract per Phase 5
- Catalog #292 per-deliberation Assumption-Adversary discipline if council escalation needed
- CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE" non-negotiable: harvest within 24h via `tac.deploy.modal.harvest_outcomes`

## OPERATOR-DECISION MATRIX

- **PROCEED**: operator authorizes $1-3 envelope; Codex executes Phase 1-6
- **PROCEED-CHEAPEST-ONLY**: operator authorizes only Phase 3 smoke ($1-3) without committing to full Phase 4-6 measurement on 3 archives; Codex returns smoke verdict + operator re-decides
- **DEFER**: Codex creates the prototype helper (Phase 2 build only; $0) without dispatch; operator decides later
- **DEFER-PENDING-FRONTIER-MOVE**: Codex waits for other in-flight work to land before this probe

## EXIT CRITERIA

- [ ] Phase 1: predecessor-probe consultation logged
- [ ] Phase 2: `tools/build_a1_per_pattern_vq_vae_inverter_prototype.py` lands
- [ ] Phase 3: smoke recipe + driver land per Catalog #270/#244 + smoke fires per Catalog #167
- [ ] Phase 4: 3-archive measurement report at `experiments/results/a1_specialized_inverter_<utc>/`
- [ ] Phase 5: Provenance-wrapped result rows per Catalog #323
- [ ] Phase 6: probe outcome registered to `.omx/state/probe_outcomes.jsonl`
- [ ] Memory entry `feedback_a1_specialized_per_pattern_vq_vae_inverter_prototype_landed_<YYYYMMDD>.md` per Catalog #229/#287 discipline

## SISTER COORDINATION

- TOP-2 (`codex_routing_directive_top2_f4_summary_512_per_pattern_inverter_prototype_20260518.md`) + TOP-3 (`codex_routing_directive_top3_f5_resblock_512_per_pattern_inverter_prototype_20260518.md`) use the same framework on different feature spaces; TOP-1's empirical anchor unlocks them
- DYNAMIC PER-CANDIDATE COMPOSITION FRAMEWORK subagent in-flight on disjoint scope
- G1 IMMEDIATE-EXECUTION routing (commit `8ebea02ef`) Codex actively executing; G1 authority-upgrade routing (commit `ecaa1c471`) queues 6-phase canonical Provenance integration

## OPERATOR-FACING NOTE

This routes TOP-1 of the SYSTEMATIC RECLAIMABILITY 5-vector reclamation set. Per CLAUDE.md "Forbidden premature KILL without research exhaustion": A1-SPECIALIZED is RECLAIMABLE via per-pattern distilled binary — this empirical probe IS the canonical research-exhaustion path. $1-3 envelope is the cheapest TOP-3 path; positive result unlocks F4 + F5 sister directives (TOP-2 + TOP-3 same framework, different feature spaces, $2-5 each).

— Main-Claude 2026-05-18 (per SYSTEMATIC RECLAIMABILITY `4480d9b14` + A1 binary distillation `0701c323b` TOP-1 ranking)
