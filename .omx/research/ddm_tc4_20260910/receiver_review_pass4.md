# TC4 receiver thread correction review — clean 1/2

Reviewer `/root/tc4_review`, 2026-09-10. Reviewed receiver SHA-256 `057e6deb8e600df9569b3acda8ae4f8cec9621ed09e98ec928e5e99b3493d73c` and the staged runtime. No receiver/scorer launched.

**CLEAN.** The inherited `f26_inflate._configure_device` explicitly requires four CPU threads. The prior review missed this call-contract mismatch; the actual failed smoke exposed it before token decoding. Replacing only `num_threads=4` back to `1` reproduces the prior reviewed source hash `5671759b…0cd0a6c`, proving the correction is limited to the host probe argument.

Verified all prior receipt and failed-log bytes/hashes in `move41/RECEIVER_CPU_THREAD_REBIND.json`. Only the producer fact differs between prior and new bindings. `CANDIDATE.construction_binding` preserves the original binding; candidate identity equals the original staged identity and the rebind receipt's unchanged identity. This is explicit host-orchestration provenance correction, not rewritten evidence of a successful smoke.

Generated-runtime review: `runtime/tc4_fast.py` is exactly the reviewed algorithm slice with its relative import, SHA-256 `98217c1e6865f71645c47e4a16e6035b244cc4a8991bb72fdcfd090f14287340` (4154 bytes). `runtime/tc4_maps.py` is exactly the reviewed map source with relative imports, SHA-256 `85c12f953f9a96d413a602d3d0feeb1e5e5d96bbc01750ba22f8ef3767c44fe0` (7779 bytes). Both compile; research CLI and filesystem routing do not enter the fast runtime export. The residual reader diff contains only TC4 rider selection, mixer dispatch preserving `tc3` observe behavior, and its report label. Its SHA-256 is `b7525c86a723ced468cacda23393cc8cb12101edb46a0ad34ce674beb638d807`. No actionable issue found in this correction and generated-runtime review.
