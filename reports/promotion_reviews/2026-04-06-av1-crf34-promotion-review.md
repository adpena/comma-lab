# AV1 CRF34 promotion review

## candidate

- Run id: `robust_current-av1-524x394-crf34-promoted-cpu-2026-04-06`
- Config: `524x394 / libsvtav1 / preset0 / crf34 / film-grain22 / bicubic / unsharp`
- Prior honest floor: `2.20` at `920,457` bytes (`crf33`)

## hypothesis before the run

- Change only one axis: `crf 33 -> 34`.
- Expected effect: lower bytes with a small distortion increase.
- Expected upside: roughly `0.01` to `0.03` score improvement **if** pose and seg distortions rose only modestly.

## measured result

- Candidate run: `2.19` at `864,456` bytes
- Canonical default-config regression: `2.19` at `864,455` bytes
- Byte delta vs prior floor: `-56,002` bytes (`-6.084%`)
- Pose delta vs prior floor: `+0.00213315`
- Seg delta vs prior floor: `+0.00013581`
- Score delta vs prior floor: `-0.0100`

## official-formula check

Using the contest formula

`100 * segnet_distortion + 25 * rate + sqrt(10 * posenet_distortion)`

with:
- seg = `0.00570503`
- pose = `0.10903914`
- rate = `0.0230242`

gives `2.1903260806708915` which is consistent with the reported `2.19` after normal rounding.

## current_workflow vs rule_faithful

- current_workflow: `2.19` at `864,455` bytes
- rule_faithful estimate: `2.2146293679221243` at `900954` bytes
- separation remains explicit and was preserved during the review

## contest-rule / path review

- Track A remains the only explicitly non-rule-faithful lane
- Track B uses stock ffmpeg codec/decode paths with no heavy decoder-side model
- inflator still emits explicit `rgb24` rawvideo bytes on the AV1 path
- archive packaging, inflation, and evaluation all completed successfully on the upstream scorer path

## bug-audit focus

The review specifically challenged these bug classes:
- raw byte layout / pixel format
- colorspace / range / conversion path
- frame count / ordering / parity-sensitive behavior
- geometry / even-dimension handling
- branch-specific behavior differences across x265, AV1, and ROI paths

At this stage, the repaired AV1 rgb24 path remains the key confirmed issue. No new blocker was found in the promotion evidence itself.

Non-blocking caveats recorded from the deeper ffmpeg-path review:
- ROI experimental branches remain x265-centric and do not yet cleanly honor live AV1 codec selection
- a dedicated pre-scorer frame-count / geometry smoke gate has now been added and the live 2.19 floor passed it
- colorspace / range drift still deserves a dedicated audit

## reflection

The hypothesis held. The distortion terms got slightly worse, but the byte reduction was large enough to more than pay for it. This suggests the local AV1 operating point still had a little additional compression headroom beyond `crf33` before task distortion dominated again.

## decision

**PROMOTE**

Reason:
- scorer-backed candidate beat the prior floor
- canonical default-config regression reproduced the win
- packaging views remained explicit
- no rule-faithful or evaluator-path blocker was introduced by the config change itself
