# L5 v2 PacketIR Section Entropy Matrix Landing - 2026-05-16

## Scope

Built a planning-only L5 v2 PacketIR section-entropy matrix over real PR106
Format0C and Format0D archives. This records both oracle context floors and
charged PCR1 prototype rows so unpriced entropy floors do not become score
claims.

## Code changes

- `tools/pr106_entropy_floor_probe.py`
  - Added Format0C magicless PacketIR unwrapping.
  - Added Format0D stacked PacketIR sidecar custody fields.
  - Added HDM9 decoder expansion through the existing HDM9->HDM8 proof helper.
  - Added HLM3 length-elided latent expansion through the existing HLM3->HLM2
    proof helper.
- `src/tac/packet_compiler/pr106_context_recode.py`
  - Added magicless PacketIR parsing for context-recode source views.
  - Added adaptive no-static-table context-range prototype rows so the next
    L5 v2 PacketIR path is measured as integrated bytes, not an unpriced floor.
- `tools/build_l5_v2_packetir_section_entropy_matrix.py`
  - New read-only matrix builder over canonical PR106 candidate specs.
  - Emits planning-only JSON/Markdown with archive custody, section floors,
    charged static prototype rows, adaptive prototype rows, blockers, and no
    score/promotability authority.
- `tools/operator_briefing.py`
  - Surfaces the latest static and adaptive section-entropy matrix in L5 v2
    frontier readiness.

## Artifacts

- `.omx/research/l5_v2_pr106_format0c_entropy_floor_probe_20260516_codex.json`
- `.omx/research/l5_v2_pr106_format0c_entropy_floor_probe_20260516_codex.md`
- `.omx/research/l5_v2_pr106_format0d_entropy_floor_probe_20260516_codex.json`
- `.omx/research/l5_v2_pr106_format0d_entropy_floor_probe_20260516_codex.md`
- `.omx/research/l5_v2_packetir_section_entropy_matrix_20260516_codex.json`
- `.omx/research/l5_v2_packetir_section_entropy_matrix_20260516_codex.md`

## Result

For both Format0C and Format0D:

- Oracle context floors on decoded section streams are large but unpriced.
- Charged PCR1 prototype rows are not rate-positive after model overhead.
- Adaptive no-static-table prototype rows are also not rate-positive yet, but
  they are close enough to be a runtime-integration target: best observed
  `latents_and_sidecar_brotli`, order 2, `delta_bytes_vs_source_section=1`.
- Matrix result: `profiled_candidate_count=2`, `prototype_row_count=12`,
  `rate_positive_prototype_row_count=0`, `adaptive_prototype_row_count=4`,
  `rate_positive_adaptive_prototype_row_count=0`.
- Best charged prototype observed in this matrix was still byte-negative:
  `latents_and_sidecar_brotli`, order 2, `delta_bytes_vs_source_section=58284`.

## Interpretation

This falsifies the naive static higher-order context table over section bytes as
an immediate byte-saving PacketIR transform for Format0C/0D. The signal is not
that L5 v2 is dead; it is that L5 v2 should not chase unpriced context-floor
mirages. The next useful L5 v2 work is a lower-overhead section transform or an
adaptive/runtime-integrated coder whose model cost is amortized or derivable
from already shipped structure.

The adaptive rows narrow that next step: static model tables are overhead-bound,
while adaptive no-table coding is only +1 byte on the latent/sidecar section and
+41 bytes on the decoder section. It is still not a score claim, not
promotion-eligible, and still blocked on runtime decoder integration,
full-frame same-runtime parity, and exact auth eval.

## Verification

```bash
.venv/bin/python -m ruff check \
  tools/operator_briefing.py \
  tools/build_l5_v2_packetir_section_entropy_matrix.py \
  src/tac/packet_compiler/pr106_context_recode.py \
  tools/pr106_entropy_floor_probe.py \
  src/tac/tests/test_operator_briefing.py \
  src/tac/tests/test_pr106_context_recode_l5_matrix.py \
  src/tac/tests/test_pr106_entropy_floor_probe.py

PYTHONPATH=src:. .venv/bin/pytest \
  src/tac/tests/test_operator_briefing.py \
  src/tac/tests/test_pr106_context_recode_l5_matrix.py \
  src/tac/tests/test_pr106_entropy_floor_probe.py -q
```

Both passed on 2026-05-16.
