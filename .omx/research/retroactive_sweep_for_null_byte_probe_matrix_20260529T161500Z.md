# Retroactive sweep for NULL-BYTE PROBE MATRIX canonical equation #26 IN-DOMAIN context expansion (Slot MM)

**Date:** 2026-05-29T16:15:00Z
**Lane:** `lane_slot_mm_null_byte_probe_matrix_canonical_equation_26_in_domain_context_expansion_cross_substrate_sweep_20260529`
**Triggered by:** canonical equation #26 anchor APPEND-ONLY event `slot_mm_cross_substrate_null_byte_probe_matrix_canonical_eq_26_in_domain_classification_20260529` per Catalog #348 + Slot DD operator-routable #2 (Wave N+48 audit RE-RUN against expanded lesson set)
**Catalog #348 4-field contract:**

## 1. Bug-class symptom signature

Anchors / verdicts that landed under canonical equation #26 IN-DOMAIN context taxonomy BEFORE the cross-substrate matrix classification overlay was registered (2026-05-29 16:30 UTC).

Search signature:
- `procedural_codebook_from_seed_compression_savings_v1` anchor rows with `inputs.in_domain_context` containing any of `pr106_format0d` / `pr107_apogee` substrings
- Sister anchor rows with predicted ΔS magnitudes ~-0.011 attributed to non-HNeRV codec families
- Canonical anti-pattern `procedural_codebook_misapplication_to_non_in_domain_codec_family_v1` matches (if any)

## 2. Pre-fix window

**Before:** 2026-05-29T16:30:00Z (Slot MM canonical equation #26 anchor_appended event)
**After:** 2026-05-29T16:30:00Z onward — NEW IN-DOMAIN candidates (`pr106_format0d_latent_score_table_zero_padded_regions` + `pr107_apogee_cd1_decoder_zero_padded_regions`) are operator-decision-pending per Catalog #344 protocol

## 3. Historical KILL / DEFER / FALSIFY search results

```bash
# Search for prior canonical equation #26 anchors referencing pr106_format0d or pr107_apogee:
.venv/bin/python -c "
import json
from pathlib import Path
events = []
with Path('.omx/state/canonical_equations_registry.jsonl').open() as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get('equation_id') == 'procedural_codebook_from_seed_compression_savings_v1':
            events.append(row)
for evt in events:
    payload = evt.get('equation_payload') or evt
    for anchor in (payload.get('empirical_anchors') or []):
        inputs = anchor.get('inputs') or {}
        ctx = inputs.get('in_domain_context', '')
        if 'pr106' in ctx or 'pr107' in ctx or 'format0d' in ctx or 'apogee' in ctx:
            print('PRIOR ANCHOR:', anchor.get('anchor_id'), 'context=', ctx)
"
```

**Empirical result at 2026-05-29T16:30:00Z**: **0 prior anchors** referenced pr106_format0d or pr107_apogee codec families in canonical equation #26 IN-DOMAIN context taxonomy. Slot MM is the FIRST anchor to identify these codec families as NEW IN-DOMAIN candidates.

**No historical KILL / DEFER / FALSIFY verdicts** in `.omx/state/probe_outcomes.jsonl` for canonical equation #26 + pr106_format0d or pr107_apogee contexts (Slot MM is the first probe outcome for this context).

**No sister anchor row** in the canonical equation #26 registry with predicted ΔS magnitudes ~-0.011 attributed to non-HNeRV codec families (the latest anchor `wave_n37_cross_substrate_pr101_hnerv_family_methodology_saturation_20260528` explicitly anchored at HNeRV-family methodology saturation, NOT non-HNeRV codec extension).

## 4. Per-finding RE-EVAL priority assignment

| Finding | RE-EVAL priority | Action |
|---|---|---|
| `pr106_format0d_latent_score_table_zero_padded_regions` NEW IN-DOMAIN candidate | **HIGH** | Operator-decision-pending per Catalog #344 protocol; 3-anchor expansion paired-CUDA RATIFICATION per Contrarian binding revision (est $0.30) |
| `pr107_apogee_cd1_decoder_zero_padded_regions` NEW IN-DOMAIN candidate | **HIGH** | Operator-decision-pending per Catalog #344 protocol; 3-anchor expansion paired-CUDA RATIFICATION per Contrarian binding revision (est $0.30) |
| Cross-substrate Dykstra-feasibility assumption (codec-family orthogonality) | **MEDIUM** | Operator-routable Dykstra alternating-projections smoke per Catalog #372; verifies aggregate predicted ΔS -0.021862 OR sub-additive |
| Byte-mutation smoke per Catalog #105 + #139 for pr106 + pr107 null bytes | **MEDIUM** | Per AssumptionAdversary unwind path; empirically verify null bytes have ZERO score effect when mutated |
| Sister codec families with N=1 anchors (BELOW canonical N>=3 threshold) | **LOW** | Documentation-only; canonical Catalog #344 protocol already imposes operator-decision-pending discipline |

---

**Sister-extinction architecture per CLAUDE.md "Bugs must be permanently fixed AND self-protected against"**:
- Catalog #344 enforces canonical equation memo-reference; Slot MM canonical anchor satisfies this gate
- Catalog #359 enforces no residual-hybrid misapplication; Slot MM in_domain_context explicitly avoids residual-hybrid patterns
- Catalog #318 enforces no master-gradient raw-byte-authority; Slot MM does NOT propose mutations (classification overlay only)
- Catalog #341 enforces Tier A canonical-routing markers; Slot MM classification matrix carries `predicted_delta_adjustment=0.0` + `promotable=False` + `axis_tag="[predicted]"`
- Catalog #356 enforces per-axis decomposition; Slot MM classification matrix carries `per_axis_null_fractions` per anchor

**No new Catalog # gate required** per Catalog #299 quota brake under 400 (current count 382; sister-extinction architecture via existing canonical surfaces preferred per CLAUDE.md "Beauty, simplicity, and developer experience").
