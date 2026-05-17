# Canonical Provenance — Developer Guide

> **Source:** operator NON-NEGOTIABLE 2026-05-17 verbatim *"We need to fix
> the provenance issue for all and fix it permanently and canonically and
> make it easy"*. Landing: `feedback_provenance_canonical_fix_meta_class_extinction_landed_20260517.md`.
> Catalog #323 STRICT preflight gate.

## TL;DR

Every score-claiming surface in the repo embeds a canonical `Provenance`
attestation:

```python
from tac.provenance import (
    Provenance,
    ScoreClaim,
    ProvenanceEvidenceGrade,
    build_provenance_for_archive_member,
    build_provenance_for_research_sidecar,
    audit_score_claim_dict,
)
```

The contract makes phantom-score class bugs **structurally impossible** at
the persisted-artifact-row surface.

## Why this exists

5 phantom-score class instances landed in one session (2026-05-17):

1. **Catalog #319** — fec6 1.15× autopilot reward (byte-identity)
2. **Catalog #321** — pr101_state_dict 0.477 (research sidecar scored)
3. pr106_state_dict / posenet_class_sensitivity (~11.6 phantom savings)
4. All 8 VALIDATED contest archives at entropy floor (WZ-on-existing-archives)
5. **Catalog #823** — α=4.74 SUPER_ADDITIVE (SIREN byte-identity artifact)

All 5 share ONE structural cause: a number was treated as a score-claim
without canonical attestation that the bytes live in a contest archive,
the axis matches the hardware, and there's no byte-identity with another
substrate. The canonical Provenance contract enforces all three.

## Quick start patterns

### Pattern 1: contest archive member (PROMOTABLE score)

```python
from tac.provenance import (
    Provenance,
    ScoreClaim,
    ProvenanceEvidenceGrade,
    build_provenance_for_archive_member,
)

prov = build_provenance_for_archive_member(
    archive_zip_path="submissions/a1/archive.zip",
    member_name="0.bin",
    measurement_axis="[contest-CUDA]",
    hardware_substrate="linux_x86_64_t4",
    evidence_grade=ProvenanceEvidenceGrade.PROMOTABLE_EXACT_CONTEST_CUDA,
)

claim = ScoreClaim(
    score_value=0.192,
    provenance=prov,
    rationale="PR101 GOLD T4 contest-CUDA replay",
)
assert claim.contest_compliant  # auto-derived from provenance
```

### Pattern 2: research sidecar (NON-PROMOTABLE)

```python
from tac.provenance import build_provenance_for_research_sidecar

prov = build_provenance_for_research_sidecar(
    sidecar_path="experiments/results/pr101_state_dict/state_dict.pt",
    reactivation_criteria="awaiting archive member byte verification",
)
# prov.score_claim_valid is False by construction.
# A non-zero score with this Provenance is the Catalog #321 phantom-score
# class — the gate refuses it.
```

### Pattern 3: predicted (autopilot ranker, NON-PROMOTABLE)

```python
from tac.provenance import build_provenance_for_predicted

prov = build_provenance_for_predicted(
    model_id="autopilot.predicted_delta_v2",
    inputs_sha256="<sha256_of_inputs>",
)
# Always non-promotable until empirical anchor lands.
```

### Pattern 4: aggregate composition (with byte-identity auto-detection)

```python
from tac.provenance import build_provenance_aggregate

agg = build_provenance_aggregate(
    parts=[prov_a, prov_b],
    aggregation_rationale="pairwise composition_alpha for lane_g_v3 × siren",
)
# If prov_a.source_sha256 == prov_b.source_sha256, agg is auto-flagged
# INVALID_BYTE_IDENTITY_ARTIFACT per Catalog #823.
```

### Pattern 5: macOS-CPU advisory (Catalog #192)

```python
from tac.provenance import build_provenance_for_macos_cpu_advisory

prov = build_provenance_for_macos_cpu_advisory(
    archive_sha256="<sha>",
    source_path="experiments/results/lane_macos/auth_eval.json",
)
# Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #192:
# macOS-CPU is NEVER 1:1 contest-compliant.
```

## Auditing existing JSON/JSONL artifacts

```python
from tac.provenance import audit_score_claim_dict

payload = json.loads(my_artifact.read_text())
valid, blockers = audit_score_claim_dict(payload, expected_axis="[contest-CUDA]")

if not valid:
    for b in blockers:
        print(f"  - {b}")
```

For sweeping the repo:

```bash
.venv/bin/python tools/audit_provenance_compliance.py --summary
.venv/bin/python tools/audit_provenance_compliance.py \
    --report-out .omx/state/provenance_audit_$(date -u +%Y%m%dT%H%M%SZ).json
```

## Decorator: `@requires_canonical_provenance`

For any function whose return value is consumed as a score:

```python
from tac.provenance import requires_canonical_provenance

@requires_canonical_provenance()
def measure_archive_score(archive_path: str) -> ScoreClaim:
    ...
    return ScoreClaim(score_value=..., provenance=...)
```

The decorator raises `MissingProvenanceError` if the return value is None,
a bare float, or a dict/object without a `provenance` field.

## Backward-compat adapters (for legacy surfaces)

If you're integrating a legacy dataclass that doesn't yet have a
`provenance` field, use the adapter shim:

```python
from tac.provenance import contest_result_to_provenance

prov = contest_result_to_provenance(legacy_result)
# Always returns a Provenance — never raises, never returns None.
# Falls back to RESEARCH_ONLY if legacy fields can't establish a
# promotable claim.
```

Available adapters:

- `contest_result_to_provenance` → `tac.continual_learning.ContestResult`
- `cost_band_anchor_to_provenance` → cost-band JSONL rows
- `council_record_to_provenance` → `CouncilDeliberationRecord`
- `substrate_composition_row_to_provenance` → composition matrix rows
- `deliverability_proof_to_provenance` → `DeliverabilityProof`
- `wyner_ziv_layer_result_to_provenance` → `WynerZivLayerResult`
- `master_gradient_plan_to_provenance` → `OptimalPerPairTreatmentPlan`
- `modal_call_id_ledger_event_to_provenance` → Modal call_id ledger events

## Evidence grades (8 canonical values)

| Grade | Promotable | Meaning |
|---|---|---|
| `PROMOTABLE_EXACT_CONTEST_CUDA` | ✓ | CUDA axis, Linux CUDA hardware, archive member |
| `PROMOTABLE_EXACT_CONTEST_CPU` | ✓ | CPU axis, Linux CPU hardware, archive member |
| `EMPIRICAL_CPU_NON_GHA` | ✗ | CPU eval but not 1:1 GHA runner |
| `MACOS_CPU_ADVISORY` | ✗ | macOS-CPU per Catalog #192 |
| `MPS_PROXY` | ✗ | MPS noise per CLAUDE.md non-negotiable |
| `PREDICTED` | ✗ | Model prediction without empirical anchor |
| `RESEARCH_ONLY` | ✗ | Research artifact (sidecar, intermediate, etc.) |
| `INVALID_BYTE_IDENTITY_ARTIFACT` | ✗ | Catalog #823 sentinel for byte-identity false signal |

## Kind taxonomy (6 canonical values)

| Kind | When to use |
|---|---|
| `CONTEST_ARCHIVE_MEMBER` | bytes ship in an archive.zip member |
| `RESEARCH_SIDECAR` | local .pt/.npy NOT in archive (Catalog #321) |
| `DERIVED_AGGREGATE` | single value derived from one upstream Provenance |
| `PREDICTED_FROM_MODEL` | autopilot/ranker prediction |
| `ADVISORY_NON_PROMOTABLE` | macOS-CPU/MPS proxies |
| `AGGREGATE_OF_PROVENANCES` | composition of multiple Provenances (Catalog #319/#823) |

## Waivers

Same-line `# PROVENANCE_CANONICAL_WAIVED:<rationale>` in JSON files
(typically as a string-value field) accepts rows that legitimately
cannot carry full Provenance (e.g., legacy archived rows pre-migration).
File-level waivers only apply to comment-style header lines; JSON string
values are evaluated as row-level waivers.

Placeholder rationales (`<rationale>` / `<reason>`) are REJECTED so the
gate's own docstring example cannot self-waive.

## When the Catalog #323 STRICT gate is flipped

Initial wire-in is WARN-ONLY at landing (2026-05-17) due to baseline
violations across `.omx/state/` and result artifacts. After the
2026-05-17 score-key synonym hardening, the Catalog #323 preflight count
is 544 warn-only rows and the operator audit reports 196 artifact-level
violations. Strict-flip happens when the
operator-routed backfill sweep brings the live count to 0 (or each row
has been waived/migrated/quarantined per CLAUDE.md "Strict-flip
atomicity rule").

## Sister gates (preserved as defense-in-depth)

- **Catalog #287** `check_empirical_claims_have_evidence` — docstring tags
- **Catalog #249** `check_no_misleading_device_named_output_directories` — filename surface
- **Catalog #319** `check_substrate_wyner_ziv_reweight_has_deliverability_proof` + sister — autopilot consumer
- **Catalog #321** `check_no_phantom_wyner_ziv_savings_from_research_sidecar` — prober artifact
- **Catalog #185** `check_strict_flipped_catalog_entries_have_live_count_zero` — META-meta drift

These remain STRICT at their own surfaces. Catalog #323 is the umbrella
that catches the META-class pattern they each address at a different surface.

## See also

- `src/tac/provenance/` — canonical package
- `tools/audit_provenance_compliance.py` — operator audit tool
- `src/tac/tests/test_provenance_*.py` — 107 tests pinning all invariants
- `src/tac/tests/test_check_323_canonical_provenance.py` — 24 gate tests
