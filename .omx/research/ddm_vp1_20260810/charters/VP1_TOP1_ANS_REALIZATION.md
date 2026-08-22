# VP1 top-1 stub — realize retained ANS on the public PR130 receiver

**Disposition:** QUEUED-WITH-A-FIRE-ORDER, folded into the active `ddm_ai1` owner; never spawn in
parallel. The public ANS code is landed; terminal unchanged-q4 replay/evaluation remains.
**Axis before evaluation:** scorer-free byte/receiver proof. **Score claim:** false.

## Mission

Consume the retained 114,860 B ANS stream through the landed public path, preserve the legacy Range
control, terminalize the unchanged-q4 archive, decode twice, and run the already-owned n600 evaluator
path only after exact receiver and runtime closure. The measured rate value is
`−2,120 B = −0.001411620980619003 S` if decoded frames remain identical.

## Inputs and pins

- PR130 archive: 191,052 B, SHA-256
  `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`.
- Retained ANS: `/Volumes/VertigoDataTier/pact/ddm_dt1_20260809/retained/ans_n600.bin`,
  114,860 B, SHA-256
  `a0b18dc0803ef541d3eb265bba5380f7aa067593f6af584b0891ded5bdd74488`.
- Measurement receipt: `.omx/research/ddm_dt1_ans_decode_wallclock_gate_20260809.md` @
  `0a0f402564d6ba45e3cc36835539d0e307bb036e`.
- Temporal measurement/consumer semantics: `.omx/research/ddm_tm1_20260809/TM1_FINDINGS.md` @
  `362bd74c70490d7b111447a5c0a342a8e3834e70`.
- Existing mixed public receiver reference: `.omx/research/ddm_cx2_20260809/CX2_FINDINGS.md` @
  `442e0d593c7635da77963c4d2d50719d0838768a`.
- Landed ANS receiver implementation: `46c7b85219` plus duplicate-launch guard `caa8eef4d8`.
- Temporal-capable real receiver reference: `.omx/research/ddm_cp2_20260810/CP2_FINDINGS.md` @
  `58d270898002cde052b4ad34506b14984db06d49`.

## Acceptance

- Exact 117,964,800-token reconstruction and empty ANS terminal state.
- Two public receiver decodes produce byte-identical raw output.
- Final archive bytes and SHA-256 are retained; no projected archive substitutes.
- Wall time remains below 1,800 seconds on the measured axis before any promotion.
- n600 result reports Seg, Pose, rate, total, GT decoder, hardware, and immutable archive hash.

## Ownership and trigger

Owner: active `ddm_ai1`. Consumer store: `/Volumes/VertigoDataTier/pact/ddm_ai1_20260809/`.
Fire trigger: the existing AI1 lane remains claimed and its input hashes match. If AI1 has already
terminalized this work, this stub is FOLDED and must not fire.
