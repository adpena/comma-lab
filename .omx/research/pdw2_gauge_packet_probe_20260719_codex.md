# PDW2 gauge-fixed generator-only packet probe (#553)

**Date:** 2026-07-19  
**Lane:** `lane_pdw2_gauge_packet_probe_20260719`  
**Posture:** `research_only=true`; local format measurement only  
**Verdict scope:** packet format plus declared native-float32 receiver arithmetic; not a
spatial/RGB receiver, through-`R` result, archive, score, launch, promotion, or family verdict

## Outcome first

**Pointer delta: exactly zero.** The pointer remains
`0.1910828242 [contest-CPU Linux x86_64]`. No scorer, archive evaluation, launch, paid dispatch,
or pointer mutation occurred.

**One-line verdict:** the additive gauge-fixed format passes the five local gates: the measured
n600 packet is `138` raw / `133` Brotli-q11 bytes with all 20 float32 margin coefficients, and the
partition-only positive-scale quotient is `134` raw / `122` Brotli-q11 bytes; both reproduce the
frame-195 native-float32 class-0/class-1 tie after strict parse-back. The required legal spatial/RGB
receiver is still absent, so the Nielsen stop rule fires: **stop packet polishing and keep the
feature-field/RGB pullback open**.

Every emitted target and receipt remains literally
`TARGET_ONLY_VS_REALIZATION_NON_EQUIVALENT`.

## Measured packet result

The byte authority is
`.omx/research/pdw2_gauge_packet_probe_20260719_receipt.json`, SHA-256
`eac796b86ee5081a6d5fb97441966c0d621a60b8dae193c35dfda603df12c5ad`.

| form | exact layout | raw | Brotli q11 | strict compressed parse-back | authority |
|---|---:|---:|---:|---|---|
| PDW2 margin-preserving | `12 header + 10 ids + 80 coefficients + 36 edges` | **138 B** | **133 B** | byte-identical | **MEASURED [byte-anchor]** |
| PDP2 partition-only | `12 header + 10 ids + 76 coefficients + 36 edges` | **134 B** | **122 B** | byte-identical | **MEASURED [byte-anchor]** |

The input is the measured n600-adjacency `PDW1`, `338 B`, SHA-256
`84a49d802dc5bd9c416013fd71bc6f08655a2f3c23c249374469a4dc4d8ee275`. PDW2
ships zero tie-locus bytes; normals and offsets are derived only after parse-back. No coefficient
uses float16 or other lossy quantization. PDP2 omits one normalized pivot scalar and stores its
index/sign inside the existing header metadata; it does **not** preserve margin magnitude.

Against 338 raw target bytes, the reductions are 200 B and 204 B. From the frozen score law these
are only **DERIVED target-rate equivalences** of `1.331717906e-6 d_seg` and
`1.358352264e-6 d_seg`, respectively, or `0.00013317179` and `0.00013583523` score units. They are
not archive savings because no spatial receiver carries the target to pixels.

## Receiver arithmetic and the frame-195 gate

Encoding subtracts the first canonical class's affine row and stores `(K-1)(d+1)=20` float32
differences. Decoding reconstructs a deterministic zero-sum common-affine gauge, then performs
multiply, ordered reduction, bias addition, pair subtraction, and first-max tie-breaking in native
float32. This arithmetic order is part of the wire contract.

That reconstruction is load-bearing. Directly treating the reference row as numerical zero after
parse-back makes class 1 positive by an ULP-class amount on the frame-195 fixture. The implemented
zero-sum reconstruction instead produces an exact float32 tie:

| form | class 0 | class 1 | pair difference | argmax |
|---|---:|---:|---:|---:|
| margin | `7.366013050079346` | `7.366013050079346` | `0.0` | `0` |
| partition-only | `1.8631844520568848` | `1.8631844520568848` | `0.0` | `0` |

This reproduces the exact diagnostic instance without a video-specific branch. The algebraically
collapsed normal/offset is still derived as a certificate, but at ULP-class margins the receiver
verdict is the subtraction of the two declared receiver scores; quotienting and finite-precision
evaluation do not commute under arbitrary reassociation.

## Five-condition gate

1. **PASS:** fixed reference-class affine gauge stores 20 float32 margin coefficients; PDP2 stores
   the separate 19-scalar positive-scale quotient.
2. **PASS:** tie normals/offsets are derived after parse-back under declared float32; 180 redundant
   PDW1 tie bytes are absent.
3. **PASS:** both pre-existing ordinary parity formulations and the exact frame-195 diagnostic are
   executable fixtures. The measured frame tie selects class 0 exactly.
4. **PASS:** encode/decode/re-encode is byte-identical for both forms. The decoder rejects bad magic,
   invalid counts/pivot, truncation, every trailer, non-finite/negative-zero coefficients, and
   noncanonical edges. There is no serialized tie field left to corrupt.
5. **PASS:** code, dataclass, receipt, and memo preserve
   `TARGET_ONLY_VS_REALIZATION_NON_EQUIVALENT`.

Focused verification: `43 passed`; ruff, `py_compile`, and `git diff --check` are clean. Three
bounded self-review passes were clean, within the authority's cap of five. The test suite includes
the categorical survivor-2 composition assertion: adding the same affine action to all class rows
leaves canonical PDW2 bytes and derived tie loci byte-identical.

## Stop, sibling convergence, and apparatus

The local packet construction meets the `<=138 B` raw margin threshold, but the spatial/RGB
receiver is absent. Per the pre-registered stop rule, no entropy-polish or alternative coder ladder
was attempted. The sibling `arith_selfcomp_rate_coders` arm owns coder-family measurements; this arm
uses Brotli q11 only because the authority explicitly requires the parse-back-counted reference.

- **Equation leg:** the measured anchor payload below is prepared for
  `segnet_head_affine_gauge_quotient_v1`; MAIN, not this branch, must append it.
- **DAG leg:** #553 format gate passes, then stops at legal spatial receiver + through-`R` closure.
- **DSL leg:** none; this research-only format probe adds no launch flag or live actuator.
- **Sensitivity/Pareto/bit allocator/autopilot:** no row is admitted until exact receiver-closed
  `(Delta bytes, Delta d_seg, Delta d_pose)` custody exists.
- **Storage/resumability:** no long run or bulk artifact was created. The sacred
  `experiments/results/levelset_n600_witness_20260717T113932Z/` directory was not modified.

## Canonical equation anchor payload for MAIN

MAIN should instantiate this payload as `EmpiricalAnchor` and call
`update_equation_with_empirical_anchor("segnet_head_affine_gauge_quotient_v1", anchor, ...)` only
after reviewing the branch diff and receipt hash.

```json
{
  "anchor_id": "pdw2_gauge_fixed_format_probe_138_134_20260719",
  "measurement_utc": "2026-07-19T10:39:11Z",
  "inputs": {
    "K": 5,
    "d": 4,
    "adjacency_edges": 9,
    "input_pdw1_bytes": 338,
    "input_pdw1_sha256": "84a49d802dc5bd9c416013fd71bc6f08655a2f3c23c249374469a4dc4d8ee275",
    "receiver_arithmetic": "native float32 fixed-reference decode plus deterministic zero-sum gauge reconstruction",
    "coefficient_quantization": "none beyond required float32"
  },
  "predicted_output": {
    "margin_raw_bytes_max": 138,
    "margin_scalar_count": 20,
    "partition_raw_bytes": 134,
    "partition_scalar_count": 19
  },
  "empirical_output": {
    "margin": {
      "raw_bytes": 138,
      "brotli_q11_bytes": 133,
      "raw_sha256": "93c0d3320e6673aed1975426a6c8c1bbc41475f295ea62b357ad7a6bf9427568",
      "strict_parseback": true
    },
    "partition_only": {
      "raw_bytes": 134,
      "brotli_q11_bytes": 122,
      "raw_sha256": "c42be295ac47ca1b33efafeb7c33259a6ae8806477ac5060cbb20a299f25b874",
      "strict_parseback": true
    },
    "frame195_native_f32": {
      "margin_exact_tie": true,
      "partition_exact_tie": true,
      "argmax": 0,
      "video_specific_exception": false
    },
    "verdict": "MEASURED_FORMAT_GATE_PASS_STOP_SPATIAL_RECEIVER_ABSENT"
  },
  "residual": 0.0,
  "source_artifact": ".omx/research/pdw2_gauge_packet_probe_20260719_receipt.json",
  "measurement_method": "strict PDW1 parse; gauge-fixed encode; Brotli q11 compress/decompress; strict PDW2 parse/re-encode; ordinary fixtures plus exact frame-195 native-f32 receiver fixture",
  "empirical_verification_status": "VERIFIED_VIA_EMPIRICAL_ANCHOR",
  "provenance": {
    "artifact_kind": "research_sidecar",
    "canonical_helper_invocation": "tac.provenance.builders.build_provenance_for_research_sidecar",
    "captured_at_utc": "2026-07-19T10:39:11Z",
    "composed_from": [],
    "contest_archive_member_name": "",
    "contest_archive_zip_path": "",
    "evidence_grade": "research_only",
    "hardware_substrate": "macos_arm64",
    "measurement_axis": "[byte-anchor]",
    "promotion_eligible": false,
    "rejection_reason": "legal spatial/RGB receiver parse-back plus measurement through R on exact candidate bytes",
    "score_claim_valid": false,
    "source_path": ".omx/research/pdw2_gauge_packet_probe_20260719_receipt.json",
    "source_sha256": "eac796b86ee5081a6d5fb97441966c0d621a60b8dae193c35dfda603df12c5ad"
  }
}
```

## Reproduction

Run from repository root; this reads only the frozen head and committed diagnostic.

```bash
PYTHONPATH=src /Users/adpena/Projects/pact/.venv/bin/python - <<'PY'
import brotli, hashlib, json
import numpy as np
from pathlib import Path
from tac.boundary_math.power_diagram_witness import (
    affine_head_to_power_diagram, decode_pdw1, decode_pdw2, encode_pdw1, encode_pdw2,
    gauge_fixed_assign_f32, gauge_fixed_pair_tie_value_f32, gauge_fixed_scores_f32,
    pdw1_to_pdw2, read_frozen_segmentation_head,
)
sha = lambda data: hashlib.sha256(data).hexdigest()
weight, bias = read_frozen_segmentation_head("/Users/adpena/Projects/pact/upstream/models/segnet.safetensors")
edges = ((0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(3,4))
head = affine_head_to_power_diagram(weight, bias, adjacency=edges)
pdw1 = encode_pdw1(head.target)
print("PDW1", len(pdw1), sha(pdw1))
for partition_only in (False, True):
    raw = encode_pdw2(pdw1_to_pdw2(head.target, partition_only=partition_only))
    compressed = brotli.compress(raw, quality=11)
    parsed = decode_pdw2(brotli.decompress(compressed))
    print(partition_only, len(raw), sha(raw), len(compressed), sha(compressed), encode_pdw2(parsed) == raw)
diagnostic = json.loads(Path(".omx/research/v10_power_diagram_frame195_diagnostic_20260718.json").read_text())
frame_target = decode_pdw1(bytes.fromhex(diagnostic["frozen_target"]["pdw1_hex"]))
point = np.asarray([diagnostic["reproduction"]["rank4_quotient"]], dtype=np.float32)
for partition_only in (False, True):
    parsed = decode_pdw2(encode_pdw2(pdw1_to_pdw2(frame_target, partition_only=partition_only)))
    scores = gauge_fixed_scores_f32(point, parsed)
    print(partition_only, scores[0, 0], scores[0, 1],
          gauge_fixed_pair_tie_value_f32(point, parsed, 0, 1)[0],
          gauge_fixed_assign_f32(point, parsed)[0])
PY
```

## Stores consulted

- `docs/operating_manual_craft_handoff.md`
- `.omx/research/nielsen_infogeo_crosswalk_20260719_codex.md`
- `.omx/research/categorical_spectrum_crosswalk_20260719_codex.md`
- `.omx/research/power_diagram_witness_20260718.md`
- `.omx/research/v10_power_diagram_frame195_diagnostic_20260718.json`
- `.omx/state/canonical_equations_registry.jsonl`
- `src/tac/canonical_equations/f32_receiver_arithmetic_law_20260719.py`
- `src/tac/canonical_equations/seg_rate_breakeven_and_head_gauge_laws_20260719.py`

This memo follows `docs/operating_manual_craft_handoff.md`: result first, labels remain attached to
numbers, the ULP failure mode is exposed rather than averaged away, and the negative is scoped only
to the missing spatial receiver.

## MAIN landing requirement

This branch is **not landed authority**. MAIN must review the complete base-to-head diff, rerun the
focused tests and reproduction command, verify the receipt hash, and only then merge and append the
prepared equation anchor through the canonical locked helper.
