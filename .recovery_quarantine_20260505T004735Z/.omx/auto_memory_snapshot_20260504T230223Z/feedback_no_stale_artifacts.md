---
name: No Stale Artifacts — Full Pipeline Consistency Required
description: Changing encode config without retraining causes distribution shift. DX must enforce consistency.
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
CRITICAL LESSON (2026-04-12): Changed BT.601 + brightness shift + chroma smooth on a BT.709-trained postfilter. Score regressed from 1.33 to 2.15. PoseNet doubled.

**Root cause:** Distribution shift. The postfilter was trained on BT.709 CRF-34 encoded video. Changing encode parameters at compress time without retraining means the postfilter sees input from a different distribution than it was trained on.

**Why:** Any encode-time change (CRF, color matrix, resolution, film grain) shifts the input distribution. The postfilter can only correct artifacts it was trained to recognize. New artifacts from a different encode config are unknown to the model.

**How to apply:**
- NEVER change encode config without either (a) retraining the postfilter or (b) proving the change is invisible to the postfilter
- DX must enforce config consistency: store a "config fingerprint" (hash of encode params) alongside the checkpoint. At inflate time, verify the archive was encoded with matching config.
- Test each exploit INDIVIDUALLY before stacking
- Always A/B test against the exact baseline config that produced the current best score
- "Theoretically free" is NOT experimentally free until measured
