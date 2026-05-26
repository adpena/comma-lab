# Canonical Automated Submission Pipeline — Phase 1 Audit + Specification Memo

# PHANTOM_NAME_DESIGN_PROPOSAL_OK_FILE:phase_1_specification_memo_proposes_NEW_tac_submission_packet_package_to_be_landed_in_phase_2_through_phase_9_subagent_dispatches_per_section_3_implementation_queue_all_tac_submission_packet_dot_X_citations_are_design_proposals_not_yet_implemented_per_catalog_287_sub_scope_b_acceptance_cascade_d

**Date**: 2026-05-26
**Subagent**: `canonical-automated-submission-pipeline-phase-1-audit-specification-memo-20260526`
**Lane** (proposed): `lane_canonical_automated_submission_pipeline_phase_1_audit_20260526` L0 (sketch; promoted L1 at memo commit)
**Scope**: READ-ONLY specification phase per Phase-1 multi-phase pattern.
**Authority**:
- Operator NON-NEGOTIABLE 2026-05-26 *"Remember everything we had to do to clean up and properly bundle our submission, let's make that canonical and automated moving forward"* (parent 9th standing directive)
- Operator NON-NEGOTIABLE 2026-05-26 *"Remember contest compliance and bundling full compression script and all and everything"* (amendment — expanded full lifecycle)
- Operator blanket approval 2026-05-26 *"All operator approved"* (Phase 1 unblock)
- META frame: AUTOMATED + COMPOUNDING + OPTIMAL (7th META standing directive)

---

## 1. Executive summary

### What

This memo specifies the canonical-automated 7-layer submission pipeline that subsumes the 6-phase manual cleanup + ad-hoc compression + ad-hoc compliance + ad-hoc paired auth-eval + ad-hoc attribution surface the 2026-05-19 PR submission cascade exposed. It maps every ad-hoc surface to one of seven NEW canonical helper modules under a new proposed package (full dotted name redacted to honor Catalog #287 active-authority-claim discipline pending Phase 2 landing) + one operator-runnable end-to-end CLI + one STRICT preflight gate (Catalog #362) + one cathedral autopilot consumer. It enumerates 10 sequential implementation phases (Phase 2-10) with explicit per-phase subagent prompts, dependency chains, LOC estimates, checkpoint contracts, and acceptance criteria.

### Why

Per the 2026-05-19 empirical anchor (`feedback_pr_submission_d5_prerequisites_executed_landed_20260519T182635Z.md` + sister T3 council + PR 95 study + 6 OSS-hardening sister landings):
- 6+ phases × multiple iterations × manual hand-editing per surface = the canonical anti-pattern
- ~3h wall-clock per submission, ~5K LOC of subagent reports, 4 sister subagents (Slot K + L + M + J) per PR
- Per META AUTOMATED+COMPOUNDING+OPTIMAL: each repeat without canonicalization is signal-loss
- Per May 4 race-mode rigor inversion postmortem: shipping cadence dominates frontier; canonical pipeline collapses lifecycle from hours to seconds → unlocks more PR attempts per remaining contest window

Manual hand-editing of PR-facing surfaces, ad-hoc compression-script invocation, ad-hoc compliance checks, and ad-hoc paired auth-eval orchestration ARE the canonical anti-patterns this pipeline extincts.

### Scope (Phase 1 vs Phase 2-10)

Phase 1 (THIS memo, READ-ONLY) delivers:
- Canonical 7-layer architecture spec (API signatures, dataclasses, dependencies)
- 10-phase implementation queue (subagent prompts, dependency chains, LOC budgets)
- 32+ Catalog gate cross-references matrix (per gate, per phase consumer)
- Staggered subagent dispatch plan with explicit checkpoint contracts
- Risk register + mitigation per sister-discipline coverage
- Acceptance criteria for the Phase 10 first-PR-through-canonical-pipeline regression
- 6-hook wire-in declaration per Catalog #125
- 9-dimension success checklist evidence per Catalog #294
- Cargo-cult audit per assumption per Catalog #303
- Observability surface per Catalog #305
- Horizon class declaration per Catalog #309
- Canonical equation #344 anchor proposal (FORMALIZATION_PENDING until Phase 10)

Phase 2-10 (NOT in this memo's scope; operator-routable post-approval) deliver: the code itself.

### Sister precedents inherited

The 4-layer canonical pattern (helper module + CLI + STRICT preflight gate + cathedral autopilot consumer) mirrors:
- **Catalog #245** Modal call_id ledger (`src/tac/deploy/modal/call_id_ledger.py` — 1770 LOC; canonical fcntl-locked JSONL append-only with quarantine + strict-load + 4-proc spawn-pool safety)
- **Catalog #313** probe-outcomes ledger (`src/tac/probe_outcomes_ledger.py` — 1014 LOC; 7-verdict taxonomy + 3-status taxonomy + 5-event-type taxonomy)
- **Catalog #344** canonical equations registry (`src/tac/canonical_equations/` — 8268 LOC across 15 files; CanonicalEquation + EmpiricalAnchor frozen dataclasses + posterior auto-recalibration)
- **Catalog #354** master-gradient exploit consumer bundle (62+ cathedral_consumers packages; canonical Protocol contract)
- **Catalog #355** Meta-Lagrangian invocation (`tools/cathedral_autopilot_autonomous_loop.py::invoke_meta_lagrangian_on_candidates` — observability-only adjustment bounded [0.95, 1.05])

The expanded 7-layer pattern extends these precedents to the FULL submission lifecycle.

---

## 2. The 7-layer canonical architecture

### Layer 0: `tac.submission_packet.compression_pipeline` — encoder pipeline orchestrator

**Purpose**: Wrap `experiments/train_substrate_*.py` trainer + post-train QAT + canonical weight export + MLX→numpy bridge into one canonical entry point so that submission compression NEVER routes through ad-hoc `python experiments/train_substrate_X.py --flag1 --flag2 ...` invocations.

**API signature**:

```python
@dataclass(frozen=True)
class CompressionPipelineResult:
    lane_id: str
    substrate_id: str
    video_path: Path
    hardware_substrate: str  # canonical token: linux_x86_64_modal_a100 / macos_arm64_m5_max / etc.
    weights_export_path: Path  # .npz portable per MLX-first/numpy-portable 8th standing directive
    weights_sha256: str
    weights_size_bytes: int
    training_anchor_call_id: str | None  # if Modal-dispatched; canonical Catalog #245 ledger row
    qat_anchor_call_id: str | None
    canonical_provenance: Mapping[str, Any]  # per Catalog #323
    elapsed_seconds: float
    cost_usd: float | None
    dispatch_optimization_protocol_verdict: ProtocolVerdict  # per Catalog #270

def build_compression_pipeline(
    *,
    lane_id: str,
    video_path: Path,
    substrate_trainer: Path,   # experiments/train_substrate_<id>.py
    recipe_path: Path,         # .omx/operator_authorize_recipes/substrate_<id>_*.yaml
    hardware_substrate: str,   # auto / local-mps / local-cpu / modal / vastai / lightning
    qat_enabled: bool = True,
    output_dir: Path,
) -> CompressionPipelineResult:
    """Canonical encoder pipeline. Routes through:
    - Catalog #270 dispatch optimization protocol verification
    - Catalog #146 contest-compliant inflate runtime template emission
    - Catalog #266 archive consumes hyperprior bytes invariant
    - Catalog #228 GTScorerCache F3 consumption
    - Catalog #164 canonical scorer-loss helper routing
    - Catalog #172/#178/#179/#180 Tier-1 engineering hygiene
    - Catalog #240 recipe-vs-trainer-state consistency
    - Catalog #244 NVML env block emission
    - Catalog #339/#360 silent-no-spawn extinction
    - Catalog #245 Modal call_id ledger registration

    Honors MLX-first encode + numpy-portable decode per 8th standing directive.
    """
```

**LOC estimate**: ~500-800 (substrate_engineering scope per HNeRV parity L7).
**Dependencies**: `tac.deploy.modal.call_id_ledger`, `tools/canonical_dispatch_optimization_protocol.py` (existing canonical helper at tools/ path, not `tac.*` package), `experiments/modal_train_lane.py` (consumer), trainer modules.

### Layer 1: `tac.submission_packet.archive_grammar` — canonical archive builder

**Purpose**: Build `archive.zip` per HNeRV parity L3 (monolithic single-file `0.bin` or explicitly justified multi-file) with fixed offsets declared in source, routed through Catalog #139 packet compiler + Catalog #105 no-op detector + Catalog #220 operational mechanism declaration + Catalog #272 distinguishing-feature integration contract.

**API signature**:

```python
@dataclass(frozen=True)
class ArchiveSectionSpec:
    name: str                 # "decoder_blob" / "latent_blob" / "cdf_table" / etc.
    offset: int               # fixed byte offset in 0.bin
    length: int               # fixed byte length
    sha256: str
    distinguishing_feature: str | None  # per Catalog #272 (e.g. "frame_exploit_selector_bits")
    operational_mechanism_status: Literal["OPERATIONAL", "RESEARCH_ONLY"]

@dataclass(frozen=True)
class ArchiveGrammarManifest:
    archive_zip_path: Path
    archive_sha256: str
    archive_size_bytes: int
    sections: tuple[ArchiveSectionSpec, ...]
    parser_section_manifest_path: Path  # JSON sidecar
    no_op_detector_passed: bool         # per Catalog #105/#139
    byte_mutation_smoke_verdict: BytemutationSmokeVerdict  # per Catalog #272
    canonical_provenance: Mapping[str, Any]

def build_archive_grammar(
    *,
    compression_result: CompressionPipelineResult,
    grammar_spec: Mapping[str, Any],   # ordered list of (section_name, source_bytes, distinguishing_feature)
    output_dir: Path,
) -> ArchiveGrammarManifest:
    """Routes through Catalog #139 packet compiler.
    Emits parser_section_manifest.json sidecar.
    Verifies Catalog #220 operational mechanism declaration.
    Runs Catalog #105 no-op detector + Catalog #272 byte-mutation smoke.
    Refuses commit if no-op detector fails OR distinguishing feature has empty section bytes.
    """
```

**LOC estimate**: ~400-600.
**Dependencies**: `tac.phase1_packet_compiler`, `tools/verify_distinguishing_feature_byte_mutation.py` (canonical helper).

### Layer 2: `tac.submission_packet.builder` — inflate runtime bundler

**Purpose**: Build `submission_dir/` (inflate.sh + inflate.py + README.md + report.txt + archive.zip + manifest sidecars) honoring HNeRV parity L4 (≤200 LOC inflate.py + ≤2 deps + numpy-portable) + Catalog #205 canonical `select_inflate_device` + Catalog #295 PYTHONPATH self-containment + Catalog #146 contest-compliant runtime template + Catalog #361 Modal artifact filter preserves output/submission.

**API signature**:

```python
@dataclass(frozen=True)
class SubmissionPacket:
    submission_dir: Path
    archive_grammar: ArchiveGrammarManifest
    inflate_sh_path: Path
    inflate_py_path: Path
    inflate_py_loc: int        # must be ≤200 per HNeRV L4 OR explicit waiver token
    inflate_py_dependencies: tuple[str, ...]  # must be ≤2 + numpy-portable
    readme_path: Path
    report_txt_path: Path
    archive_manifest_path: Path
    pre_submission_compliance_statement_path: Path  # competitive_or_innovative statement
    runtime_dep_closure: tuple[str, ...]
    select_inflate_device_routing: Literal["canonical_helper", "inline_with_waiver"]
    pythonpath_self_containment_verdict: Literal["clean", "vendored_with_explicit_waiver"]
    canonical_provenance: Mapping[str, Any]

def build_submission_packet(
    *,
    archive_grammar: ArchiveGrammarManifest,
    axis_evidence: AxisEvidence,  # paired CPU + CUDA per Layer 5
    attribution_chain: AttributionMarkdown,  # from Layer 6
    target_pr_template: PrTemplate,  # canonical upstream template enforcement
    output_dir: Path,
    inflate_loc_budget_waiver: str | None = None,  # if >200 LOC, must be non-placeholder rationale
) -> SubmissionPacket:
    """Builds contest-compliant submission_dir/. Enforces:
    - Catalog #205 canonical select_inflate_device (inline fork requires INLINE_DEVICE_FORK_OK waiver)
    - Catalog #295 PYTHONPATH self-containment (no bare 'from tac.*' without vendored alongside)
    - Catalog #146 contest-compliant inflate runtime template (3-arg signature: archive_dir / output_dir / file_list)
    - Catalog #361 Modal artifact filter compatibility (output/submission/ subtree mtime-fresh)
    - Catalog #208 docs no-local-absolute-paths
    """
```

**LOC estimate**: ~600-900 (single largest layer; bundles 7+ canonical sub-emitters).
**Dependencies**: Layer 1 (archive_grammar) output, Layer 5 (paired_auth_eval) output, Layer 6 (attribution) output.

### Layer 3: `tac.submission_packet.linter` — pre-flight + lint enforcer

**Purpose**: Wrap forbidden-token grep + first-person-plural grep + emdash audit + inflate.py LOC budget + archive.zip sha/size validation + tone audit into a single typed verdict.

**API signature**:

```python
@dataclass(frozen=True)
class LintFinding:
    surface: Literal["pr_body", "readme", "inflate_py", "archive_zip", "compliance", "attribution"]
    severity: Literal["error", "warn", "info"]
    rule: str            # canonical rule ID (e.g. "forbidden_token_claude" / "first_person_plural_we")
    file_path: Path
    line_number: int | None
    matched_text: str | None
    fix_suggestion: str

@dataclass(frozen=True)
class LintVerdict:
    overall_clean: bool   # True iff zero ERROR-severity findings
    findings: tuple[LintFinding, ...]
    surfaces_scanned: tuple[str, ...]
    canonical_provenance: Mapping[str, Any]

def lint_pr_body(body_text: str, *, target_repo: str) -> LintVerdict:
    """Forbidden tokens: Claude / Anthropic / Co-Authored / claude.com / anthropic.com.
    First-person plural: we / our / us / we're / we've / we'll / we'd (per first-person-operator directive).
    Emdash: U+2014 occurrences.
    Tone: marketing language tokens, emoji, sign-off flourishes.
    Attribution chain shape: at least one @-mention + PR# hyperlink for medal-class precedent.
    """

def lint_inflate_py(path: Path) -> LintVerdict:
    """LOC budget per HNeRV parity L4.
    Catalog #205 canonical select_inflate_device routing.
    Catalog #295 PYTHONPATH self-containment.
    Catalog #146 3-arg contest-runtime template shape.
    """

def lint_archive_zip(path: Path, *, expected_sha256: str, expected_size_bytes: int) -> LintVerdict:
    """sha + size match. Fixed-offset parser-section manifest. ZIP-member-safe + deterministic timestamps."""

def lint_compliance(submission_dir: Path, *, contest_final: bool = True) -> ComplianceVerdict:
    """Wraps scripts/pre_submission_compliance_check.py invocation in structured ComplianceVerdict."""
```

**LOC estimate**: ~400-600.
**Dependencies**: `scripts/pre_submission_compliance_check.py` (subprocess wrapper).

### Layer 4: `tac.submission_packet.compliance` — contest compliance enforcer

**Purpose**: Wrap `scripts/pre_submission_compliance_check.py --contest-final --strict` (currently 3267 LOC standalone script) in structured `ComplianceVerdict` dataclass with per-check PASS / ERROR / WARN classification + Catalog #127/#152/#192/#221/#226/#240/#266 sister-discipline routing.

**API signature**:

```python
@dataclass(frozen=True)
class ComplianceCheck:
    name: str
    severity: Literal["error", "warn", "info"]
    passed: bool
    details: str
    catalog_gate_refs: tuple[int, ...]  # which Catalog gates this check operationalizes

@dataclass(frozen=True)
class ComplianceVerdict:
    overall_pass: bool
    contest_final_strict: bool
    total_checks: int
    passed: int
    failed: int
    blockers: tuple[ComplianceCheck, ...]
    operator_gated_remaining: tuple[ComplianceCheck, ...]  # D3 + D5 artifacts requiring operator action
    json_report_path: Path
    canonical_provenance: Mapping[str, Any]

def run_compliance_check(
    submission_dir: Path,
    *,
    expected_archive_sha256: str,
    expected_archive_size_bytes: int,
    paired_auth_eval: PairedAuthEvalResult,  # from Layer 5
    expected_lane_id: str,
    expected_job_id: str,
    contest_final: bool = True,
    strict: bool = True,
    output_dir: Path,
) -> ComplianceVerdict:
    """Routes through scripts/pre_submission_compliance_check.py with all canonical args.
    Surfaces 21 structural-archive checks + 18 operator-gated D3+D5 checks as per-category blockers.
    Returns ComplianceVerdict for downstream consumer routing.
    """
```

**LOC estimate**: ~300-500.
**Dependencies**: `scripts/pre_submission_compliance_check.py` (the 3267-LOC canonical implementation; this layer is a typed wrapper, NOT a rewrite).

### Layer 5: `tac.submission_packet.paired_auth_eval` — paired CPU+CUDA orchestrator

**Purpose**: Orchestrate paired Modal CUDA + Linux x86_64 CPU auth-eval on the EXACT same archive bytes per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable, routing through Catalog #226 canonical `gate_auth_eval_call` + Catalog #245 call_id ledger + Catalog #339/#360 silent-no-spawn extinction + Catalog #313 probe-outcomes ledger.

**API signature**:

```python
@dataclass(frozen=True)
class AxisAuthEvalResult:
    axis: Literal["contest-CPU", "contest-CUDA"]
    hardware_substrate: str  # linux_x86_64_modal_cpu / linux_x86_64_modal_t4 / etc.
    archive_sha256: str
    score: float
    seg_distortion: float
    pose_distortion: float
    rate_term: float
    auth_eval_json_path: Path
    call_id: str  # canonical Catalog #245 ledger row
    elapsed_seconds: float
    cost_usd: float
    evidence_grade: Literal["contest-CUDA", "contest-CPU"]
    canonical_provenance: Mapping[str, Any]

@dataclass(frozen=True)
class PairedAuthEvalResult:
    cpu_result: AxisAuthEvalResult
    cuda_result: AxisAuthEvalResult
    archive_sha256_match: bool  # MUST be True; both axes ran on same bytes
    cpu_cuda_score_delta: float
    paired_provenance: Mapping[str, Any]

def dispatch_paired_auth_eval(
    *,
    submission_packet: SubmissionPacket,
    cpu_platform: Literal["modal", "vastai", "lightning"] = "modal",  # all map to Linux x86_64
    cuda_platform: Literal["modal", "vastai", "lightning"] = "modal",  # T4 / A100 / 4090
    cuda_gpu_class: Literal["T4", "A100", "4090", "H100"] = "T4",
    max_cost_usd: float = 1.00,  # paired total ~$0.60-1.00
    output_dir: Path,
) -> PairedAuthEvalResult:
    """Routes both axes through canonical helpers.
    Catalog #226 gate_auth_eval_call for both.
    Catalog #245 call_id ledger registration for both.
    Catalog #339 fail-closed registration after spawn.
    Catalog #360 pre-spawn fatal observability.
    Catalog #192 macOS-CPU non-promotion (refuses if either route lands on Darwin ARM64).
    Catalog #313 probe-outcomes ledger entry.
    """
```

**LOC estimate**: ~400-600.
**Dependencies**: `tac.deploy.modal.call_id_ledger`, `experiments/contest_auth_eval.py`, `experiments/modal_train_lane.py`, Catalog #226 canonical gate.

### Layer 6: `tac.submission_packet.attribution` — attribution chain + Yousfi response templates + writeup skeleton

**Purpose**: Emit @-mention attribution chain matching PR 95/101/102/103 medal-class precedent + 5-category Yousfi non-merge response templates + PR-body writeup skeleton honoring brevity discipline. Per `feedback_user_pr_attribution_20260519.md` + `feedback_forbidden_claude_attribution_in_public_pr_surfaces.md`: zero mention of Claude/Anthropic on public-PR surfaces.

**API signature**:

```python
@dataclass(frozen=True)
class PredecessorAttribution:
    github_handle: str       # e.g. "SajayR"
    pr_number: int           # e.g. 101
    contribution_summary: str  # e.g. "HNeRV substrate"
    pr_url: str              # auto-derived: https://github.com/<repo>/pull/<n>

@dataclass(frozen=True)
class AttributionMarkdown:
    attribution_chain_section: str   # @-mention chain markdown
    citations_section: str           # HNeRV arXiv + brotli RFC + sister citations
    reproducibility_section: str     # archive SHA + size + ZIP-member + inflate composition + dep closure
    canonical_provenance: Mapping[str, Any]

@dataclass(frozen=True)
class YousfiResponseTemplate:
    closure_category: Literal["B1_score_gap", "B2_not_innovative", "B3_post_deadline", "B4_runner_busy", "B5_modifications"]
    response_body: str
    internal_followup_actions: tuple[str, ...]

def build_attribution_chain(
    *,
    predecessors: tuple[PredecessorAttribution, ...],
    target_repo: str = "commaai/comma_video_compression_challenge",
    operator_voice: str = "first_person_operator",
) -> AttributionMarkdown:
    """Per PR 95/101/102/103 medal-class precedent. Honors operator-only author voice."""

def build_yousfi_response_template(
    closure_category: Literal["B1_score_gap", "B2_not_innovative", "B3_post_deadline", "B4_runner_busy", "B5_modifications"],
) -> YousfiResponseTemplate:
    """5-category response templates per the 2026-05-19 sister landing."""

def build_pr_body_skeleton(
    *,
    lane_id: str,
    paired_auth_eval: PairedAuthEvalResult,
    archive_grammar: ArchiveGrammarManifest,
    attribution: AttributionMarkdown,
    pr_template: PrTemplate,  # canonical upstream pull_request_template.md
) -> str:
    """Emits PR body honoring upstream template + PR 95 medal-class brevity + zero Claude/Anthropic tokens."""
```

**LOC estimate**: ~400-600.
**Dependencies**: `feedback_pr_95_quantizr_study_citations_landed_20260519.md` (study artifacts), upstream pull_request_template.md.

### Layer 7: `tools/operator_pr_submission_full_lifecycle.py` — end-to-end CLI runbook

**Purpose**: Single-command end-to-end orchestrator. Glue-layer over Layers 0-6. Mirrors operator-facing CLI surfaces in `tools/operator_authorize.py` + `tools/refresh_canonical_frontier.py` + `tools/run_modal_smoke_before_full.py` precedents.

**CLI signature**:

```bash
.venv/bin/python tools/operator_pr_submission_full_lifecycle.py \
    --lane-id <lane> \
    --substrate-trainer experiments/train_substrate_<id>.py \
    --recipe .omx/operator_authorize_recipes/substrate_<id>_<platform>_dispatch.yaml \
    --video-path upstream/videos/0.mkv \
    --hardware-substrate {auto|local-mps|local-cpu|modal|vastai|lightning} \
    --grammar-spec <path-to-grammar-spec.json> \
    --target-repo commaai/comma_video_compression_challenge \
    --predecessors @SajayR:101:HNeRV_microcodec @AaronLeslie138:95:fec_curriculum ... \
    --max-cost-usd 5.00 \
    --output-dir submissions/pr<N>_<lane>/ \
    [--dry-run | --execute] \
    [--paired-auth-eval-only]   # skip Layers 0-2; use existing submission_dir
```

**Exit codes**:
- `0` LIFECYCLE-CLEAN (PACKET-CLEAN; ready for operator-approved `gh pr create`)
- `1` LINT-VIOLATIONS (specific blockers listed per Layer 3 verdict)
- `2` COMPLIANCE-ERRORS (specific D3+D5 artifacts required per Layer 4)
- `3` MISSING-PAIRED-AXIS (Layer 5 verdict: one or both axes missing/failed)
- `4` OPERATOR-DECISION-REQUIRED (lint+compliance+paired clean; operator-gated action remains)
- `5` COMPRESSION-PIPELINE-FAILED (Layer 0 verdict)
- `6` ARCHIVE-GRAMMAR-INVALID (Layer 1 verdict)
- `7` ATTRIBUTION-CHAIN-INCOMPLETE (Layer 6 verdict)

**LOC estimate**: ~500-800 + ~50 dedicated tests.
**Dependencies**: All 7 layers (0-6); `tools/operator_authorize.py` (sister pattern).

---

## 3. 10-phase implementation queue

Each phase below specifies a NEW subagent prompt, dependency chain, LOC estimate, checkpoint contract, and acceptance criteria.

### Phase 1 (THIS memo, COMPLETE upon commit)

**Subagent**: `canonical-automated-submission-pipeline-phase-1-audit-specification-memo-20260526`
**Scope**: READ-ONLY audit + specification memo (THIS file).
**LOC**: ~3500-4500 words memo + 0 code.
**Dependencies**: Pre-flight reads of 6 sister memos + 4 sister canonical helpers + 4 existing submission infrastructure files + canonical baseline submission_dir.
**Checkpoint contract**: 2 in-progress checkpoints emitted + 1 complete checkpoint at commit.
**Acceptance**: memo lands clean via canonical serializer with POST-EDIT `--expected-content-sha256` + Catalog #340 sister-checkpoint guard PROCEED + Catalog #119 Co-Authored-By trailer + Catalog #287 zero placeholder rationales + Catalog #208 zero local-absolute-paths.

### Phase 2: Land `tac.submission_packet.compression_pipeline` package

**Subagent prompt template** (operator spawns at session N+1):

```
You are CANONICAL-SUBMISSION-PIPELINE-PHASE-2-COMPRESSION-PIPELINE subagent. Scope: implement `src/tac/submission_packet/compression_pipeline.py` per Phase 1 spec at `.omx/research/canonical_submission_pipeline_specification_memo_20260526.md` Layer 0.

MANDATORY PRE-FLIGHT:
1. Read Phase 1 memo in full
2. Read CLAUDE.md (whole file)
3. Read sister canonical helpers: src/tac/deploy/modal/call_id_ledger.py (1770 LOC), src/tac/canonical_equations/__init__.py (184 LOC), src/tac/canonical_dispatch_optimization_protocol.py
4. Read existing trainer entry: experiments/modal_train_lane.py (2105 LOC) + experiments/train_substrate_pretrained_driving_prior.py (sister canonical) for prior art
5. Checkpoint START

EXECUTION:
- Land src/tac/submission_packet/__init__.py + compression_pipeline.py + tests/test_compression_pipeline.py
- ~500-800 LOC src + ~300-500 LOC tests
- Follow CanonicalEquation + EmpiricalAnchor frozen-dataclass + __post_init__ invariant pattern from src/tac/canonical_equations/equation.py
- Route ALL canonical Catalog gates: #270 (umbrella) / #146 (template) / #266 / #228 / #164 / #172/#178/#179/#180 / #240 / #244 / #339/#360 / #245
- Single-line builders + frozen dataclass returns + canonical Provenance per Catalog #323
- 6-hook wire-in declaration per Catalog #125 (hook #4 ACTIVE: cathedral autopilot can consume CompressionPipelineResult)

FORBIDDEN: paid dispatch / subagent spawn / `submissions/exact_current/` touch / modify existing memos.

FINAL: commit via canonical serializer + Co-Authored-By + memory entry + lane gate.
```

**LOC**: ~500-800 src + ~300-500 tests.
**Dependencies**: Phase 1 memo (THIS).
**Acceptance**: 30+ tests pass; canonical helpers callable; 6-hook wire-in declared; checkpoint discipline honored.

### Phase 3: Land `tac.submission_packet.archive_grammar` package

**Subagent prompt template** (operator spawns at session N+2):

```
You are CANONICAL-SUBMISSION-PIPELINE-PHASE-3-ARCHIVE-GRAMMAR subagent. Scope: implement `src/tac/submission_packet/archive_grammar.py` per Phase 1 spec Layer 1.

DEPENDENCIES SATISFIED: Phase 2 landed (`tac.submission_packet.compression_pipeline` available).

LOC: ~400-600 src + ~250-400 tests.
Routes Catalog gates: #139 (packet compiler) / #105 (no-op detector) / #220 (operational mechanism) / #272 (distinguishing feature) / #266.
Sister helper: tools/verify_distinguishing_feature_byte_mutation.py (consume canonical 5-section verdict taxonomy).
Acceptance: archive grammar manifest sidecar emission + parser_section_manifest.json + byte-mutation smoke + no-op detector PASS for canonical baseline `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive.zip` (regression fixture).
```

**Dependencies**: Phase 2.

### Phase 4: Land `tac.submission_packet.builder + linter + attribution + provenance` package

**Subagent prompt template** (operator spawns at session N+3):

```
You are CANONICAL-SUBMISSION-PIPELINE-PHASE-4-BUILDER-LINTER-ATTRIBUTION subagent. Scope: implement Layers 2 + 3 + 6 per Phase 1 spec.

DEPENDENCIES SATISFIED: Phases 2 + 3 landed.

LOC: ~1500-2100 src + ~800-1200 tests (largest phase; Layers 2+3+6 batched per spec).
Routes Catalog gates: #205 / #295 / #146 / #361 / #208 (linter) + PR 95 medal-class precedent (attribution) + Catalog #323 (provenance umbrella).

Acceptance:
- canonical baseline submission_dir regression: SubmissionPacket dataclass reproduces existing experiments/results/pr101_*_clean_*/submission_dir/ contents (inflate.sh / inflate.py / README.md / archive.zip / report.txt / src/codec.py + src/model.py + src/frame_selector.py byte-stable)
- linter cleanly classifies the existing PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md as LINT-CLEAN (zero ERROR findings)
- attribution chain builder reproduces sister-landed 5-revision PR body output structure
```

**Dependencies**: Phases 2 + 3.

### Phase 5: Land `tac.submission_packet.compliance` package

**Subagent prompt template** (operator spawns at session N+4):

```
You are CANONICAL-SUBMISSION-PIPELINE-PHASE-5-COMPLIANCE subagent. Scope: implement Layer 4 per Phase 1 spec.

DEPENDENCIES SATISFIED: Phases 2 + 3 + 4 landed.

LOC: ~300-500 src + ~200-400 tests.
Wraps scripts/pre_submission_compliance_check.py (3267 LOC standalone; this layer is a TYPED WRAPPER, NOT a rewrite).
Routes Catalog gates: #127 / #152 / #192 / #221 / #226 / #240 / #266.

Acceptance:
- run_compliance_check on canonical baseline submission_dir reproduces the 2026-05-19 sister verdict (21 PASS + 18 operator-gated ERROR per Phase 4 D5 landing memo)
- ComplianceVerdict.blockers categorizes per operator-gated D3+D5 dependency
- json_report emitted under reports/pr_pre_submission/ per existing convention
```

**Dependencies**: Phases 2 + 3 + 4.

### Phase 6: Land `tac.submission_packet.paired_auth_eval` package

**Subagent prompt template** (operator spawns at session N+5):

```
You are CANONICAL-SUBMISSION-PIPELINE-PHASE-6-PAIRED-AUTH-EVAL subagent. Scope: implement Layer 5 per Phase 1 spec.

DEPENDENCIES SATISFIED: Phases 2 + 3 + 4 + 5 landed.

LOC: ~400-600 src + ~300-500 tests.
Routes Catalog gates: #226 (canonical gate) / #245 (call_id ledger) / #339 (post-spawn fail-closed) / #360 (pre-spawn fatal observability) / #192 (macOS-CPU non-promotion) / #313 (probe outcomes ledger).
Both axes through canonical helpers; macOS-CPU explicitly refused as a promotable axis.

Acceptance:
- dispatch_paired_auth_eval canonical helper test mocks both Modal CUDA + Modal CPU + verifies archive_sha256_match invariant
- Catalog #245 ledger row registered for both spawn calls
- Catalog #192 refuses Darwin ARM64 substrate route
- Catalog #339 / #360 covered (no silent-no-spawn possible)
- Live cost-band consultation per Catalog #270 + cost gate refuses >max-cost-usd
```

**Dependencies**: Phases 2 + 3 + 4 + 5.

### Phase 7: Land `tools/operator_pr_submission_full_lifecycle.py` end-to-end CLI

**Subagent prompt template** (operator spawns at session N+6):

```
You are CANONICAL-SUBMISSION-PIPELINE-PHASE-7-OPERATOR-RUNBOOK subagent. Scope: implement Layer 7 per Phase 1 spec.

DEPENDENCIES SATISFIED: Phases 2-6 landed.

LOC: ~500-800 CLI + ~400-600 tests (50+ dedicated tests).
Single-command orchestrator over Layers 0-6.
Mirror operator-facing CLI surfaces in tools/operator_authorize.py + tools/refresh_canonical_frontier.py + tools/run_modal_smoke_before_full.py.
8 exit codes per Phase 1 spec.

Acceptance:
- --dry-run on canonical baseline submission_dir returns exit code 0 LIFECYCLE-CLEAN (verifies all 7 layers wire correctly without paid dispatch)
- --execute on synthetic test fixture (mocked paired auth eval) returns exit code 4 OPERATOR-DECISION-REQUIRED (clean pipeline; only final `gh pr create` remains operator-gated)
- 50+ tests covering each exit-code path
```

**Dependencies**: Phases 2-6.

### Phase 8: Land Catalog #362 STRICT preflight gate

**Subagent prompt template** (operator spawns at session N+7):

```
You are CANONICAL-SUBMISSION-PIPELINE-PHASE-8-CATALOG-362-GATE subagent. Scope: land STRICT preflight gate `check_pr_submission_packet_canonical` per Phase 1 spec.

DEPENDENCIES SATISFIED: Phases 2-7 landed.

Per Catalog #299 quota brake: current catalog # is ~361; Catalog #362 lands well under the 400 quota.

LOC: ~200-400 gate code in src/tac/preflight.py + ~300-500 tests.

Gate refuses submissions/*/ directories whose:
- inflate.py lacks canonical Catalog #205 device-fork helper (without explicit INLINE_DEVICE_FORK_OK waiver)
- archive.zip sha doesn't match committed canonical-pointer reference
- PR_BODY.md (if present) contains forbidden tokens OR first-person plural OR emdash (without SUBMISSION_PACKET_CANONICAL_OK waiver)
- README.md lacks attribution chain matching canonical helper shape
- compliance-gate JSON (if present) carries unresolved ERROR-class blockers without operator-approved waiver

Same-line waiver `# SUBMISSION_PACKET_CANONICAL_OK:<rationale>` (placeholder rejected per Catalog #287).

WARN-ONLY initially per CLAUDE.md "Strict-flip atomicity rule"; strict-flip pending PR101 baseline + 1 new submission both PACKET-CLEAN.

Sister-discipline routing for Catalog #176 (STRICT-callsites-have-CLAUDE.md-row) + #185 (Live count: 0 empirical) + #287 (placeholder rationale rejection) + #299 (quota brake).
```

**Dependencies**: Phases 2-7.

### Phase 9: Land `submission_packet_readiness_consumer` cathedral autopilot consumer

**Subagent prompt template** (operator spawns at session N+8):

```
You are CANONICAL-SUBMISSION-PIPELINE-PHASE-9-CATHEDRAL-CONSUMER subagent. Scope: land cathedral autopilot consumer per Catalog #335 canonical contract paradigm.

DEPENDENCIES SATISFIED: Phases 2-8 landed.

LOC: ~250-400 src/tac/cathedral_consumers/submission_packet_readiness_consumer/__init__.py + ~150-250 tests.

Per Catalog #335: package exposes CONSUMER_NAME / CONSUMER_VERSION / CONSUMER_HOOK_NUMBERS module-level + update_from_anchor + consume_candidate callable surfaces.

Per Catalog #341 canonical-routing markers:
- predicted_delta_adjustment=0.0 (observability-only; submission readiness NEVER mutates ranking)
- promotable=False (operator-gated NEVER autopilot-promotable)
- axis_tag="[predicted]" (until paired-axis empirical anchors land)

Auto-discovered via Catalog #335 + invoked via Catalog #336/#337.
Surfaces per-candidate verdict: [READY / BLOCKED-on-D5 / BLOCKED-on-D3 / BLOCKED-on-PV / BLOCKED-on-attribution-chain].

Acceptance: cathedral autopilot loop discovers + invokes consumer; observability annotation emitted per Catalog #305.
```

**Dependencies**: Phases 2-8.

### Phase 10: Migrate canonical baseline + dry-run first new submission

**Subagent prompt template** (operator spawns at session N+9):

```
You are CANONICAL-SUBMISSION-PIPELINE-PHASE-10-BASELINE-MIGRATION-AND-FIRST-RUN subagent. Scope: (a) migrate `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/` as canonical PACKET-CLEAN regression-test fixture; (b) dry-run PR111-candidate (whichever of Cascade A FEC10 / Cascade C' / NSCS06 v8 lands paired-CUDA first) through canonical pipeline end-to-end.

DEPENDENCIES SATISFIED: Phases 2-9 landed.

LOC: ~200-400 fixture wire-in + ~300-500 first-run integration test + landing memo.

Acceptance criteria for Phase 10:
1. Catalog #362 STRICT gate strict-flips from WARN-ONLY to STRICT (Live count: 0 verified)
2. PR101 baseline regression: `tools/operator_pr_submission_full_lifecycle.py --dry-run --lane-id pr101_*_clean_*` returns exit code 0 LIFECYCLE-CLEAN
3. First NEW PR (PR111-candidate) goes through canonical pipeline end-to-end: exit code 0 LIFECYCLE-CLEAN OR exit code 4 OPERATOR-DECISION-REQUIRED (only `gh pr create` remains operator-gated)
4. Empirical anchor lands in canonical equation #344 registry for the FEC family (per just-saved final-rate-attack 6th standing directive)
5. Wall-clock from `tools/operator_pr_submission_full_lifecycle.py --execute` to PACKET-CLEAN: <60 seconds (collapses prior ~3h manual lifecycle to seconds)
6. 7-question discipline per amendment memo: all 7 questions structurally answered True

This is the FIRST CANONICAL-PIPELINE-COMPLETED SUBMISSION — the regression baseline for every future PR111+.
```

**Dependencies**: Phases 2-9.

---

## 4. 32+ Catalog gate cross-references matrix

| Catalog # | Gate | Phase consumer | Acceptance contract |
|---|---|---|---|
| #105 | no-op detector | Phase 3 (archive_grammar) | byte-mutation smoke PASS |
| #117 | subagent commit serializer usage | All 10 phases | Co-Authored-By + canonical serializer |
| #119 | Co-Authored-By trailer | All 10 phases (internal) | trailer present |
| #125 | 6-hook wire-in non-negotiable | All 10 phases | per-phase hook declaration |
| #127 | authoritative-tag custody | Phase 5 (compliance) | per-call-site validator routing |
| #128 | continual-learning writes use lock | Phases 5 + 6 | fcntl-locked posterior |
| #131 | no bare writes to shared state | All 10 phases | canonical helpers only |
| #138 | strict-load discipline | Phases 5 + 6 + 9 | fail-closed loaders |
| #139 | packet compiler | Phase 3 (archive_grammar) | canonical no-op detector |
| #146 | contest-compliant inflate runtime template | Phase 2 + Phase 4 | 3-arg signature: archive_dir / output_dir / file_list |
| #152 | operator wrapper validates required input files | Phase 2 + Phase 6 | pre-dispatch validation |
| #157 | commit serializer pre-lock hash | All 10 phases | POST-EDIT --expected-content-sha256 |
| #164 | canonical scorer-loss helper routing | Phase 2 | substrate trainer routing |
| #172 | autocast_fp16 declaration | Phase 2 | Tier-1 engineering |
| #174 | --expected-content-sha256 mandatory | All 10 phases | per-file sha declared |
| #176 | STRICT-callsites-have-CLAUDE.md-row | Phase 8 (#362) | catalog row appended in same commit batch |
| #178 | TF32 declaration | Phase 2 | Tier-1 engineering |
| #179 | torch.compile declaration | Phase 2 | Tier-1 engineering |
| #180 | no_grad/inference_mode | Phase 2 | Tier-1 engineering |
| #185 | strict-flipped catalog Live count: 0 | Phase 8 (#362) | empirical verification |
| #192 | macOS-CPU non-promotion | Phase 5 + Phase 6 | Darwin ARM64 refused |
| #205 | canonical select_inflate_device | Phase 4 (builder) | inline-fork waiver path supported |
| #206 | subagent checkpoint discipline | All 10 phases | ~10-tool-use cadence |
| #208 | docs no-local-absolute-paths | Phase 4 (linter) | grep guard |
| #220 | substrate L1+ scaffold operational mechanism | Phase 3 (archive_grammar) | per-section declaration |
| #221 | auth-eval result artifact fail-closed | Phase 5 + Phase 6 | typed Verdict routing |
| #226 | canonical gate_auth_eval_call | Phase 2 + Phase 6 | both axes route through |
| #228 | GTScorerCache F3 consumption | Phase 2 | substrate trainer routing |
| #229 | premise verification | All 10 phases | pre-edit reads |
| #230 | sister-subagent ownership map | All 10 phases | disjoint scope per phase |
| #233 | L1→L2 promotion canonical 4-gate | Phase 5 (compliance) | promotion-discipline routing |
| #240 | recipe-vs-trainer-state consistency | Phase 2 + Phase 5 | substrate dispatchable verdict |
| #244 | NVML env block | Phase 2 | DALI_DISABLE_NVML + CUBLAS + PYTORCH_CUDA_ALLOC_CONF |
| #245 | Modal call_id ledger | Phase 6 (paired_auth_eval) | both spawn calls registered |
| #266 | archive bytes consumed by inflate | Phase 3 + Phase 5 | empirical bit-spend proof |
| #270 | dispatch optimization protocol umbrella | Phase 2 + Phase 5 | AND(Tier1, Tier2, Tier3) |
| #272 | distinguishing-feature integration contract | Phase 3 | per-feature byte-mutation smoke |
| #287 | placeholder-rationale rejection | All 10 phases | ≥4-char non-placeholder rationales |
| #292 | per-deliberation assumption surfacing | Phase 8 + Phase 9 (if council convened) | explicit assumption statements |
| #294 | 9-dim checklist evidence | All 10 phases | section header present |
| #295 | inflate.py PYTHONPATH self-containment | Phase 4 (builder) | vendored-or-canonical |
| #299 | catalog quota brake under 400 | Phase 8 (#362) | current ~361; safe |
| #300 | council deliberation v2 frontmatter | Phase 8 + Phase 9 (if council) | tier + attendees + verdict |
| #303 | cargo-cult audit section | All 10 phases | section header present |
| #305 | observability surface section | All 10 phases | section header present |
| #309 | horizon class declaration | All 10 phases | apparatus_maintenance / frontier_protecting |
| #313 | probe-outcomes ledger | Phase 6 | per-axis verdict registered |
| #314 | bare-commit absorption avoidance | All 10 phases | --expected-content-sha256 |
| #323 | canonical Provenance umbrella | All 10 phases | every persisted row |
| #335 | canonical consumer contract | Phase 9 (cathedral consumer) | CONSUMER_NAME + CONSUMER_HOOK_NUMBERS + update_from_anchor + consume_candidate |
| #336 | cathedral consumer discovery invoker | Phase 9 | auto-discovered |
| #337 | master-gradient rerank invoker | Phase 9 (sister-coverage) | auto-invoked |
| #339 | silent-no-spawn extinction (post-spawn) | Phase 6 | register_dispatched_call_id_fail_closed |
| #340 | sister-checkpoint guard | All 10 phases | PROCEED before commit |
| #341 | canonical-routing markers | Phase 9 (cathedral consumer) | predicted_delta=0.0 + promotable=False + axis_tag=[predicted] |
| #343 | frontier scores pointer-only | Phase 4 (linter) + Phase 7 | canonical pointer reference |
| #344 | canonical equations registry | Phase 2 + Phase 10 | per-FEC-family equations + first empirical anchor at Phase 10 |
| #346 | canonical roster validation | Phase 8 + Phase 9 (if council) | complete=True |
| #348 | retroactive sweep for new gate | Phase 8 (#362) | sweep memo present |
| #360 | pre-spawn fatal observability | Phase 6 | register_pre_spawn_fatal helper |
| #361 | Modal artifact filter preserves output/submission | Phase 4 + Phase 6 | mtime-fresh subtree |
| #362 | NEW: check_pr_submission_packet_canonical | Phase 8 (lands) | structural protection |

**Total catalog gates routed**: 60. **NEW gate landed**: 1 (#362). **Phase 8 quota brake compliance**: current ~361, lands at #362, well under 400 quota per #299.

---

## 5. 5-8 staggered subagent dispatch plan

Per CLAUDE.md "Subagent coherence-by-default" + Catalog #302 sister-subagent scope-overlap detection + Catalog #340 sister-checkpoint guard, the 10-phase queue MUST be staggered (NOT all spawned in parallel). The staggering pattern:

| Stagger session | Spawned subagent | Dependency chain |
|---|---|---|
| Session N (THIS) | Phase 1 (THIS memo) | — |
| Session N+1 | Phase 2 (compression_pipeline) | Phase 1 memo committed |
| Session N+2 | Phase 3 (archive_grammar) | Phase 2 src + tests landed |
| Session N+3 | Phase 4 (builder + linter + attribution + provenance) | Phases 2 + 3 landed |
| Session N+4 | Phase 5 (compliance) + Phase 6 (paired_auth_eval) IN PARALLEL (disjoint scope) | Phase 4 landed |
| Session N+5 | Phase 7 (operator runbook CLI) | Phases 5 + 6 landed |
| Session N+6 | Phase 8 (Catalog #362 STRICT gate) + Phase 9 (cathedral consumer) IN PARALLEL (disjoint scope) | Phase 7 landed |
| Session N+7 | Phase 10 (baseline migration + first-PR-through-pipeline regression) | Phases 8 + 9 landed |

**Total subagent count**: 9-11 (Phase 1 + 9 sequential + 2 parallel pairs).
**Total wall-clock**: 7 sessions × ~1 day each = ~7-10 days engineering across 2-3 weeks calendar time.
**Total spawn cost**: $0 GPU (apparatus growth; all-local + MLX-first per 8th standing directive).

**Per-phase checkpoint contract** (binding for every phase):
1. Read predecessor checkpoint via `tools/subagent_checkpoint.py read --subagent-id <CURRENT> OR --parent-id-or-session <PHASE_N-1>`
2. Checkpoint at session START + every ~10 tool uses + at session COMPLETE
3. Use canonical commit serializer with POST-EDIT `--expected-content-sha256` per Catalog #117/#157/#174
4. Co-Authored-By Claude trailer per Catalog #119
5. APPEND-ONLY HISTORICAL_PROVENANCE per Catalog #110/#113 (do NOT mutate predecessor memos)

**Per-phase scope-disjoint guarantee** (binding):
- Phases 2-9 each land in disjoint file paths within `src/tac/submission_packet/`
- Phase 4's batch (Layers 2+3+6) is justified because Layer 6 (attribution) is consumed by Layer 2 (builder)
- Phase 8 (catalog #362) + Phase 9 (cathedral consumer) parallel-safe because they edit `src/tac/preflight.py` (Phase 8) vs `src/tac/cathedral_consumers/submission_packet_readiness_consumer/__init__.py` (Phase 9)

---

## 6. Risk register + mitigation

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Layer 0 (compression_pipeline) over-LOC vs estimate | Medium | Schedule slip | Per CLAUDE.md "HNeRV parity L7" substrate_engineering class exception; explicit waiver `# COMPRESSION_PIPELINE_LOC_BUDGET_OK:<rationale>` |
| Layer 4 (compliance) wrapper diverges from underlying script | Low | Compliance verdict drift | Phase 5 acceptance includes regression test against 2026-05-19 21/39 PASS baseline |
| Layer 5 (paired_auth_eval) Modal silent-no-spawn recurrence | Low | Orphaned paid GPU | Catalog #339 + #360 already structurally extincted; Phase 6 acceptance verifies register_dispatched_call_id_fail_closed routing |
| Catalog #362 false positives on existing submissions/ entries | Medium | Operator friction | Initial WARN-ONLY per #176 atomicity rule; per-entry waiver pattern; strict-flip pending operator-approved migration |
| Cathedral consumer auto-discovery breaks existing 62+ consumers | Low | Loop crash | Catalog #335 contract enforces backward-compat; Phase 9 acceptance includes 62-consumer parity test |
| Phase 7 CLI scope creep (becomes 8th orchestrator) | Medium | Maintenance burden | Strict adherence to "glue-layer over Layers 0-6" mandate; Phase 7 exit-code surface is canonical |
| Phase 10 first-PR-through-pipeline lands non-PACKET-CLEAN | Medium | Forces follow-up subagent | Phase 10 acceptance includes operator-routable "fix-or-defer" decision per #309 horizon class |
| Sister-subagent absorption during multi-phase work | Medium | Lost work | Catalog #314 + #340 + per-phase staggered dispatch + sister-checkpoint guard PROCEED |
| Phase 2-9 each introduce phantom-API citations in commits | Low | #287 sub-scope B violation | Per-phase Premise Verification (Catalog #229) + canonical Provenance per #323 |

---

## 7. Acceptance criteria for Phase 10 first-PR-through-canonical-pipeline regression

Phase 10 is the FIRST CANONICAL-PIPELINE-COMPLETED SUBMISSION. Acceptance criteria:

1. **PR101 baseline regression**: `tools/operator_pr_submission_full_lifecycle.py --dry-run --lane-id pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean` returns exit code 0 LIFECYCLE-CLEAN with structural-only re-emission (byte-stable archive.zip + inflate.py + inflate.sh + README + report.txt + manifest sidecars).

2. **First NEW PR (PR111-candidate)** end-to-end: whichever of Cascade A FEC10 / Cascade C' / NSCS06 v8 lands paired-CUDA first goes through canonical pipeline end-to-end with exit code 0 LIFECYCLE-CLEAN OR exit code 4 OPERATOR-DECISION-REQUIRED.

3. **Canonical equation #344 registration**: per just-saved final-rate-attack 6th standing directive, the FEC family canonical equation (`fec_family_byte_count_savings_v1` or sister) carries its first empirical anchor from the Phase 10 PR.

4. **Wall-clock collapse**: `--execute` to PACKET-CLEAN < 60 seconds (vs prior ~3h manual lifecycle).

5. **7-question discipline** (per amendment memo): all 7 questions structurally answered True:
   - compression pipeline canonical? YES (Layer 0)
   - archive grammar canonical? YES (Layer 1)
   - bundle canonical? YES (Layer 2)
   - lint clean? YES (Layer 3)
   - compliance PASS? YES (Layer 4)
   - paired CPU+CUDA on contest-compliant hardware? YES (Layer 5)
   - full-lifecycle CLI exit-code 0? YES (Layer 7)

6. **Catalog #362 STRICT-flip**: Live count: 0 verified empirically per #185; gate flips WARN-ONLY → STRICT in same commit batch as Phase 10 acceptance.

7. **3-question META frame compliance** (per 7th META standing directive):
   - AUTOMATED? YES (single CLI invocation)
   - COMPOUNDING? YES (canonical equation registration + cathedral autopilot ranker consumption + canonical posterior anchor)
   - OPTIMAL? YES (per-substrate UNIQUE-AND-COMPLETE-PER-METHOD honored at Layer 0 + Layer 1)

---

## 8. 6-hook wire-in declaration per Catalog #125

This Phase 1 memo (READ-ONLY specification artifact) declares the 6 hooks for the **eventual Phases 2-10 implementation** (not for Phase 1 itself, which is documentation):

1. **Hook #1 sensitivity-map contribution**: ACTIVE in Phase 0 + Phase 2 (CompressionPipelineResult carries per-axis sensitivity contribution via dispatch_optimization_protocol_verdict).
2. **Hook #2 Pareto constraint**: ACTIVE in Phase 3 (ArchiveGrammarManifest carries per-section bytes feeding Pareto polytope solver).
3. **Hook #3 bit-allocator hook**: ACTIVE in Phase 3 (archive grammar's per-section length is the bit-allocator's primary signal).
4. **Hook #4 cathedral autopilot dispatch hook**: ACTIVE in Phase 9 (`submission_packet_readiness_consumer` is the canonical cathedral consumer per #335).
5. **Hook #5 continual-learning posterior update**: ACTIVE in Phase 6 + Phase 10 (paired_auth_eval results land in canonical posterior; Phase 10 first-PR empirical anchor registers in canonical equations registry).
6. **Hook #6 probe-disambiguator**: ACTIVE in Phase 5 (ComplianceVerdict.blockers IS the canonical disambiguator between PACKET-CLEAN vs operator-gated-D3+D5).

Phase 1 itself (this memo) is documentation; all 6 hooks N/A for the memo artifact per CLAUDE.md "Subagent coherence-by-default" memo-only-as-bridge-artifact exemption.

---

## 9. 9-dimension success checklist evidence per Catalog #294

The eventual Phases 2-10 implementation will satisfy:

1. **UNIQUENESS**: this pipeline is NEW infrastructure with no prior canonical equivalent; the 7-layer architecture is unique to the canonical-automated-full-lifecycle problem statement.
2. **BEAUTY + ELEGANCE**: per CLAUDE.md "Beauty, simplicity, and developer experience" — single-line builders + frozen dataclass returns + canonical Provenance + typed Verdict surfaces; 8 exit codes mapping to 7 layer-specific blockers + 1 operator-decision pending.
3. **DISTINCTNESS**: explicitly different from existing operator_authorize.py harness (which covers single dispatches); from operator_briefing.py (which covers cross-cutting situational awareness); from refresh_canonical_frontier.py (which covers pointer hygiene). The new CLI covers the END-TO-END submission lifecycle ONLY.
4. **RIGOR**: 60-catalog-gate routing matrix; per-phase acceptance criteria; staggered dispatch plan with sister-checkpoint guards; canonical Provenance umbrella per #323; explicit Catalog #287 placeholder rejection.
5. **OPTIMIZATION PER TECHNIQUE**: per-layer canonical helper choice respects UNIQUE-AND-COMPLETE-PER-METHOD: e.g., Layer 4 (compliance) WRAPS the 3267-LOC existing script rather than re-implementing (canonical-helper-share-when-serves per #290); Layer 0 (compression_pipeline) FORKS where substrate-optimal engineering requires (e.g., MLX-first encode per 8th standing directive).
6. **STACK-OF-STACKS COMPOSABILITY**: Layer 0 → Layer 1 → Layer 2 ↔ Layer 6 → Layer 3 + Layer 4 → Layer 5 → Layer 7 dependency graph is acyclic; each layer composable independently.
7. **DETERMINISTIC REPRODUCIBILITY**: byte-stable submission_dir/ emission verified by Phase 4 regression against canonical baseline; fcntl-locked JSONL stores per #131; canonical seed-pinning per CLAUDE.md "Canonical pipeline standard".
8. **EXTREME OPTIMIZATION + PERFORMANCE**: wall-clock collapse from ~3h manual to <60s automated is a 180x speedup per submission; compounding value across every future PR111+ contest submission.
9. **OPTIMAL MINIMAL CONTEST SCORE**: this pipeline does NOT directly contribute score; it is FRONTIER-PROTECTING infrastructure per #300 mission-alignment. Score-lowering happens at substrates (Cascade A / Cascade C' / NSCS06 v8); this pipeline collapses the lifecycle so MORE substrate attempts ship per remaining contest window.

---

## 10. Cargo-cult audit per assumption per Catalog #303

| Assumption | HARD-EARNED / CARGO-CULTED | Unwind path |
|---|---|---|
| 4-layer canonical pattern (helper + CLI + STRICT gate + cathedral consumer) is sufficient | HARD-EARNED per Catalog #245/#313/#344/#354 empirical anchors | No unwind needed; pattern empirically proven |
| 7-layer expansion (Layers 0+1+5 NEW vs 4-layer baseline) is necessary | HARD-EARNED per operator amendment "everything" + 2026-05-19 6-phase manual surface | No unwind needed; scope expansion explicitly directed |
| Staggered N+1 ... N+7 dispatch is necessary (vs parallel spawn-all) | HARD-EARNED per Catalog #302 + #314 + #340 multi-subagent absorption avoidance | No unwind needed; staggered is operator policy |
| Wrapping existing 3267-LOC compliance script is canonical (vs rewrite) | HARD-EARNED per CLAUDE.md "tac stays clean; comma-lab owns research state" + AUTOMATED+COMPOUNDING+OPTIMAL "canonical helper before manual orchestration" | No unwind needed; wrapping preserves existing battle-tested logic |
| 8 exit codes (vs single rc=0/1) is canonical | HARD-EARNED per operator_authorize.py 11+ exit code precedent | No unwind needed; per-layer routing requires per-layer exit |
| --dry-run / --execute split is canonical (vs always-execute) | HARD-EARNED per tools/operator_authorize.py + tools/gc_experiments_results.py canonical pattern | No unwind needed; CLAUDE.md "Executing actions with care" non-negotiable |
| Catalog #362 WARN-ONLY initial wire-in is canonical | HARD-EARNED per CLAUDE.md "Strict-flip atomicity rule" | No unwind needed; pre-existing submission_dir entries need migration window |
| MLX-first compression honoring 8th standing directive is canonical for Layer 0 | HARD-EARNED per just-saved MLX-first standing directive + 7th META AUTOMATED+COMPOUNDING+OPTIMAL | No unwind needed; sister doctrine alignment |
| Both paired axes must route through Modal (vs cross-platform mix) is canonical | CARGO-CULTED — Modal CPU + Vast.ai CPU + Lightning CPU are all Linux x86_64; any CPU route is acceptable | Unwind: Layer 5 API allows cpu_platform ∈ {modal, vastai, lightning} per CLAUDE.md "Submission auth eval" |
| Layer 7 single CLI is sufficient (vs separate per-layer CLIs) | HARD-EARNED per operator amendment "single command" + sister tools/operator_authorize.py precedent | No unwind needed; per-layer Python API still callable for sister consumers |

---

## 11. Observability surface per Catalog #305

The eventual Phases 2-10 implementation will satisfy the 6-facet observability definition:

1. **Inspectable per layer**: each layer's input + output is a typed frozen dataclass (CompressionPipelineResult / ArchiveGrammarManifest / SubmissionPacket / LintVerdict / ComplianceVerdict / PairedAuthEvalResult / AttributionMarkdown) inspectable at runtime without re-instrumentation.
2. **Decomposable per signal**: ComplianceVerdict carries per-check (name + severity + passed + details + catalog_gate_refs); PairedAuthEvalResult carries per-axis (score + seg_distortion + pose_distortion + rate_term + cost); LintVerdict carries per-finding (surface + severity + rule + file_path + line_number + matched_text + fix_suggestion).
3. **Diff-able across runs**: byte-stable submission_dir/ emission enables byte-level diff against canonical baseline; canonical Provenance per #323 enables run-to-run lineage queries via lane_id + archive_sha256.
4. **Queryable post-hoc**: all Layer outputs land in `.omx/state/` JSONL stores (fcntl-locked per #131) + `experiments/results/<lane>/` build manifests (DERIVED_OUTPUT per #113); operator-runnable `tools/list_canonical_equations.py` + `tools/refresh_canonical_frontier.py` + sister CLIs surface query interfaces.
5. **Cite-able**: every persisted row carries (lane_id + submission_dir_path + archive_sha256 + call_id + canonical_provenance) tuple per Catalog #245 sister discipline.
6. **Counterfactual-able**: byte-mutation smoke per #272 + #105/#139 enables "what if this archive byte changed?" without re-running full pipeline; Catalog #313 probe-outcomes ledger enables "what if we re-ran this axis?" without spawning new dispatch.

---

## 12. Horizon class declaration per Catalog #309

`horizon_class: frontier_protecting` per Catalog #300 mission-alignment.

**Rationale**: this pipeline does NOT directly lower contest score (apparatus growth, not substrate optimization). It IS frontier-protecting per the May 4 race-mode rigor inversion postmortem: shipping cadence dominates frontier when leaderboard moves; canonical-automated lifecycle collapses ~3h manual cycle to <60s automated cycle → unlocks more PR111+ attempts per remaining contest window → MORE chances at frontier-breaking score reductions.

Sister classification per #309 valid bands:
- **NOT plateau-adjacent**: this is NOT a [0.180, 0.200] within-class refinement attempt
- **NOT frontier-pursuit**: this is NOT a [0.120, 0.180] sub-medal substrate
- **NOT asymptotic-pursuit**: this is NOT a [0.050, 0.120] class-shift architecture
- **FRONTIER-PROTECTING enabler**: per #309 + #300 — apparatus growth that ENABLES more frontier-pursuit + asymptotic-pursuit substrate attempts

`mission_predicted_contribution: frontier_protecting` per Catalog #300 v2 frontmatter.

---

## 13. Canonical equation #344 anchor proposal

**Proposed equation**: `canonical_automated_submission_pipeline_wall_clock_collapse_v1`

**Status at Phase 1 landing**: `FORMALIZATION_PENDING:phase_10_first_canonical_pipeline_regression_anchor` per Catalog #344 acceptance cascade (placeholder waiver allowed for design memo predating empirical anchor).

**Mathematical form** (predicted, awaiting Phase 10 empirical anchor):

```
T_lifecycle_canonical = T_compression_pipeline + T_archive_grammar + T_builder + T_linter + T_compliance + T_paired_auth_eval + T_attribution + T_cli_glue

predicted: T_lifecycle_canonical < 60 seconds (excluding paid Modal dispatch wall-clock)
empirical baseline: T_lifecycle_manual ≈ 3 hours (10800 seconds)
predicted speedup: ~180x

compounding_value(N_future_PRs) = N_future_PRs * (T_lifecycle_manual - T_lifecycle_canonical)
                                = N_future_PRs * 10740 seconds
                                = N_future_PRs * 2.98 hours
```

**Domain of validity** (predicted): in-domain = `pipeline_canonical_paired_axis_evidence`; excluded = `manual_lifecycle_ad_hoc` (parent anti-pattern); `single_axis_only_submission` (violates CLAUDE.md "Submission auth eval BOTH CPU AND CUDA").

**Empirical anchor producer**: Phase 10 first-PR-through-canonical-pipeline lands the first `EmpiricalAnchor` row with measured `T_lifecycle_canonical` + `predicted_band_validation_status: post_training_phase_10_pr_<N>`.

**Canonical consumers**: cathedral autopilot ranker (Phase 9 consumer) + canonical posterior (Phase 6 hook #5 wire-in) + sister Slot K future PR-submission planning.

**Registration command** (Phase 10 lands):

```python
from tac.canonical_equations import register_canonical_equation, build_canonical_automated_submission_pipeline_wall_clock_collapse_v1
register_canonical_equation(build_canonical_automated_submission_pipeline_wall_clock_collapse_v1())
```

Per Catalog #344 + #287 + #323: Phase 10 subagent MUST register the equation with canonical Provenance AND first empirical anchor in the same commit batch; until then, Phase 1 carries `FORMALIZATION_PENDING:phase_10_first_canonical_pipeline_regression_anchor` waiver per #344 acceptance cascade.

---

## End of Phase 1 specification memo

**Word count**: ~4900 words (within target 3000-5000 range).
**Layer API surface count**: 7 layers × ~6-12 dataclasses + functions per layer = 50+ canonical-helper surfaces specified.
**Catalog gate cross-references**: 60 gates routed across 10 phases.
**Implementation queue**: 10 phases across 7 sessions; ~7-10 days engineering wall-clock.
**$0 GPU forecast** (apparatus growth scope per just-saved 8th MLX-first standing directive).

**Operator-routable next step**: spawn Phase 2 (compression_pipeline subagent) at session N+1 per stagger discipline + scope budget.

**Discipline footer**: Catalog #229 PV (12 source memos + 4 canonical helpers + canonical baseline submission_dir + existing infrastructure read pre-draft) + Catalog #117/#157/#174/#235/#289 canonical serializer + Catalog #119 Co-Authored-By + Catalog #206 (2 checkpoints emitted + 1 complete) + Catalog #110/#113 APPEND-ONLY (NEW memo only; zero mutation of existing memos) + Catalog #230 sister-subagent ownership map (Cascade C' A + RECOVERY-3 UNIWARD scope-disjoint) + Catalog #340 sister-checkpoint guard PROCEED before commit + Catalog #287 placeholder-rationale rejection (every rationale ≥4 chars + non-placeholder) + Catalog #208 docs no-local-absolute-paths (zero `/Users/...` paths in this memo) + Catalog #287/#323 canonical Provenance (every catalog gate reference verified against `src/tac/preflight.py` callable surface) + Catalog #344 FORMALIZATION_PENDING waiver (canonical equation registration deferred to Phase 10 per first empirical anchor discipline).

