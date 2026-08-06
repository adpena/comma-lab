# ddm_et2 NEXT_IF_RESUMED

Status: fire-order 1 A/B completed and folded.

Do not fire full n600 byte-close for either projected-static rank-6 Q3 phase-field arm from this run.

- Arm E: eta `0.040337200870195794` vs bar `0.1710048742006269`, pose max `1.0356058119444502` (pose pass).
- Arm M: eta `0.04396301667875272` vs bar `0.1710048742006269`, pose max `1.389875680200799` (pose fail).
- Winner for the registered A/B is Arm E because it is pose-neutral; neither arm clears eta.

If resumed for audit or reproduction:

- Reuse parent decode and score cache under `/Volumes/VertigoDataTier/pact/ddm_et2_20260806/parent_tq1c_decode` and `/Volumes/VertigoDataTier/pact/ddm_et2_20260806/parent_score`.
- Reuse re-solved block16 offsets under `/Volumes/VertigoDataTier/pact/ddm_et2_20260806/phase_field`.
- Reuse Arm E rows at `/Volumes/VertigoDataTier/pact/ddm_et2_20260806/fire_order_1_projected_rows.jsonl`.
- Reuse Arm M rows at `/Volumes/VertigoDataTier/pact/ddm_et2_20260806/fire_order_1_m_projected_rows.jsonl`.
- Reassemble final JSON with:

```bash
PYTHONPATH=src .venv/bin/python experiments/ddm_et2_projected_phase_field.py --resume
```

Next live scientific fork, if authorized by MAIN, is not fire-order 2 from this arm. It is a separate formulation: solve inside Q3 / constrained descent on the phase-field target, with a current-vehicle pose leakage guard and its own denominator. Do not blend that with the projected-static eta measured here.
