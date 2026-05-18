# Codex routing directive: OP-SYN-1 — extend master-gradient extractor to 6 archives
# Date: 2026-05-18
# Operator: approved 2026-05-18 ("All are approved")
# Authority: Cross-stack synthesis memo .omx/research/cross_stack_synthesis_9_design_landings_unified_framework_20260518.md
#   §9 TOP-5 op-routables; OP-SYN-1 = highest-EV synthesis-level unblocker (∞ EV; $0 cost)
# Per CLAUDE.md "Frontier target" + canonical empirical-anchor discipline

## CANONICAL POINTERS (read FIRST)

1. `/Users/adpena/Projects/pact/CLAUDE.md` (FULL; especially Catalog #318 typed-operator API + #319 deliverability tiers + #322 composition_alpha + #324 post-training Tier-C validation)
2. `.omx/research/cross_stack_synthesis_9_design_landings_unified_framework_20260518.md` (§9 OP-SYN-1; §2 universal empirical anchor)
3. `tools/extract_master_gradient.py` (current state: PR101_lc_v2 Phase A LANDED; supported grammars enumerated in source)
4. `src/tac/master_gradient.py` + `src/tac/master_gradient_consumers.py` (canonical helper)
5. `.omx/state/master_gradient_anchors.jsonl` (canonical posterior)
6. `src/tac/codec/wyner_ziv_layer.py` + `src/tac/wyner_ziv_deliverability/` + `src/tac/procedural_codebook_generator/` + `src/tac/null_space_exploiter/` (consumers of multi-archive master-gradient anchors)

## EMPIRICAL ANCHOR (the bottleneck this resolves)

PR101_lc_v2 archive `f174192aeadf...` is currently the SOLE archive with per-pair fp64 master-gradient anchor. Per cross-stack synthesis §2 + §9 OP-SYN-1: ALL 9 of today's design landings + the cheap-probe wave + DP1+PR101 composition + the deterministic-byte-derivation subsystem ANCHOR on this single archive. Without 6-archive extension, downstream OPs are GATED on PR101_lc_v2 only — losing the cross-substrate-class signal.

The 6 target archives (per synthesis §9):
1. **A1** — sub-0.193 CPU anchor; current frontier; primary contest validation
2. **PR101_lc_v2** ✅ ALREADY LANDED (Phase A in Codex session 019de465 ITEM_3)
3. **PR106_format0d** — 0.20533 [contest-CUDA] frontier substrate (per Catalog #316 frontier scan)
4. **PR107_apogee** — research substrate; apogee_intN family
5. **DP1** — pretrained driving prior; canonical OOD substrate (per Catalogs #209/#210/#211/#213)
6. **fec6** — fec6 stacking-wave substrate (TIER-1 anchor wave LANDED per task #788+#882)

## DELIVERABLE

Extend `tools/extract_master_gradient.py` with 5 NEW archive-grammar parsers (PR101_lc_v2 already supported). Each new parser:
- Inherits the canonical fp64 extraction pattern from PR101_lc_v2 Phase A
- Honors Catalog #318 typed-operator API (NO raw byte-modification arrays; emit `CandidateModificationSpec`)
- Honors Catalog #327 contest-axis custody (`evidence_grade` + `axis` + `hardware_substrate` per Catalog #127)
- Writes per-pair fp64 master-gradient sidecar to `.omx/state/master_gradient_sidecars/<archive_sha256>_per_pair.pt` (gitignored; large)
- Writes aggregate row to `.omx/state/master_gradient_anchors.jsonl` per Catalog #131 fcntl-locked append-only

## ARCHITECTURE

```python
# tools/extract_master_gradient.py extension

@register_archive_grammar_parser("A1")
def parse_a1_archive(archive_path: Path) -> ArchiveGrammarContext:
    """Parse A1 archive (current sub-0.193 CPU frontier). Grammar: HNeRV-monolithic single 0.bin file."""
    # ... follows PR101_lc_v2 pattern with A1-specific section offsets

@register_archive_grammar_parser("PR106_format0d")
def parse_pr106_format0d_archive(archive_path: Path) -> ArchiveGrammarContext:
    """Parse PR106 format0d archive. Grammar: HNeRV + format0d latent score table."""
    # ... PR106 format0d has additional latent_score_table section

@register_archive_grammar_parser("PR107_apogee")
def parse_pr107_apogee_archive(archive_path: Path) -> ArchiveGrammarContext:
    """Parse PR107 apogee archive (apogee_intN family). Grammar: HNeRV + intN quantization."""

@register_archive_grammar_parser("DP1")
def parse_dp1_archive(archive_path: Path) -> ArchiveGrammarContext:
    """Parse DP1 archive (pretrained driving prior). Grammar: 162164-byte decoder blob + 15387-byte latent blob."""
    # ... DP1 has dedicated decoder_blob + latent_blob layout per Catalog #210

@register_archive_grammar_parser("fec6")
def parse_fec6_archive(archive_path: Path) -> ArchiveGrammarContext:
    """Parse fec6 stacking archive. Grammar: PR101 base + fec6 selector layer."""
```

Each parser MUST:
- Be ~100-200 LOC
- Have its own dedicated test in `src/tac/tests/test_extract_master_gradient.py` (extend existing 32 tests)
- Emit failures via canonical fail-closed pattern per Catalog #279 (no PASS-VACUOUS)
- Register via `register_archive_grammar_parser` decorator that auto-wires into the multi-archive extractor

## CLI EXTENSION

```bash
# Extract from any of the 6 supported archives:
.venv/bin/python tools/extract_master_gradient.py extract --archive-sha256 <sha> --archive-grammar A1 --archive-path <path> --output-dir .omx/state/master_gradient_sidecars/
.venv/bin/python tools/extract_master_gradient.py extract --archive-sha256 <sha> --archive-grammar PR106_format0d --archive-path <path>
.venv/bin/python tools/extract_master_gradient.py extract --archive-sha256 <sha> --archive-grammar PR107_apogee --archive-path <path>
.venv/bin/python tools/extract_master_gradient.py extract --archive-sha256 <sha> --archive-grammar DP1 --archive-path <path>
.venv/bin/python tools/extract_master_gradient.py extract --archive-sha256 <sha> --archive-grammar fec6 --archive-path <path>

# List supported grammars:
.venv/bin/python tools/extract_master_gradient.py list-grammars

# Batch extract all 6:
.venv/bin/python tools/extract_master_gradient.py extract-all --manifest experiments/results/master_gradient_6_archive_batch_20260518/manifest.json
```

## DOWNSTREAM UNBLOCKS (per synthesis §9 EV ranking)

After this lands:
- **OP-7** Direct master-gradient pose-byte hoist (cheap-probe wave) extends to 6 archives instead of just PR101_lc_v2
- **OP-2** Master-gradient pose-byte classification extension extends to 6 archives
- **OP-10** Cathedral autopilot Cascade 2 extension consumes 6-archive coverage
- **Phase 1 Fisher-precondition** Tier-A canonical helper extends Fisher conditioning verdicts to 6 archives
- **3-set Venn classifier** extends per-byte position classification to 6 archives
- **Riemannian-Newton + Tropical d_seg** consume 6-archive master-gradient as local linearization
- **DP1+PR101 composition** Path A OP-1 (OOD-similarity probe) + OP-2 (architecture-compatibility probe) consume DP1 master-gradient + PR101 master-gradient PAIRED
- **Catalog #319 DeliverabilityTier classification** extends per-archive

## DISCIPLINE

- Catalog #229 premise verification BEFORE editing (verify each archive's grammar via existing intake / sister memos)
- Catalog #117/#157/#174 commit serializer with POST-EDIT shas
- Catalog #186 catalog # claim (if new STRICT gate needed; likely not — pure extension of existing extractor)
- Catalog #206 checkpoint discipline every ~10 tool uses
- Catalog #131 fcntl-locked writes to `.omx/state/master_gradient_anchors.jsonl`
- Catalog #287 evidence-tag discipline on all numeric outputs
- Catalog #327 contest-axis authority — sidecar artifacts MUST carry axis + hardware_substrate + evidence_grade
- Catalog #313 register probe outcome via `tac.probe_outcomes_ledger.register_probe_outcome` for each archive extraction
- Catalog #314 absorption avoidance: file scope is `tools/extract_master_gradient.py` + `src/tac/tests/test_extract_master_gradient.py` + `.omx/state/master_gradient_anchors.jsonl`. Sister subagents own `.omx/research/*.md` only.

## EXIT CRITERIA

- [ ] 5 NEW archive-grammar parsers landed (A1 + PR106_format0d + PR107_apogee + DP1 + fec6)
- [ ] Tests extended (existing 32 tests + 5 × ~4 new = 50+ total) all pass
- [ ] CLI `list-grammars` returns 6 grammars
- [ ] CLI `extract-all` batch succeeds on test fixtures
- [ ] `.omx/state/master_gradient_anchors.jsonl` contains 6 rows (one per archive)
- [ ] codex_persistent_session_state row appended with `directive_executed=op_syn_1_master_gradient_six_archive_extension`
- [ ] canonical_task_status row updated 'completed' with commit_shas + test_status
- [ ] Memory entry `feedback_master_gradient_6_archive_extension_landed_20260518.md`

## OPERATOR-FACING NOTE

After this lands, the universal anchor bottleneck (PR101_lc_v2-only) is resolved. Every downstream OP across today's 9 design landings becomes parallel-executable across 6 archives instead of sequential through PR101_lc_v2. This is the synthesis's #1 ranked op-routable.

— Main-Claude 2026-05-18 (synthesis-memo-authorized routing per OP-SYN-1 EV ranking)
