---
name: GPU Lane IS the CPU Lane
description: Quantizr's insight — renderer works on both GPU and CPU. One archive serves both. CPU postfilter is a separate submission.
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
GPU lane IS the CPU lane. The neural renderer works on BOTH devices — GPU for speed (~2s), CPU as fallback (~3min). One archive.zip, one inflate.py.

**Why:** Quantizr proved this at 0.60. Their 386KB renderer runs on any device. The distinction between "CPU lane" and "GPU lane" is a false dichotomy for renderers. The only thing that changes is batch size and wall-clock time.

**How to apply:**
- The CPU postfilter (current 1.93) is a SEPARATE submission, not the "CPU lane" of the renderer submission
- The Fridrich renderer submission works on both CPU and GPU — no separate inflate paths needed
- When discussing submissions: "postfilter submission" vs "renderer submission", not "CPU lane" vs "GPU lane"
- The renderer is the ONLY path to beating Quantizr. The postfilter is insurance.
