# F26 native HPAC/RC64 receiver

This directory contains the generic C lowering and audit bundle for the F26
token receiver. The reviewed Python binding is in `experiments/`. The native
function consumes archive-derived model codes,
residual values, causal plans, and the original RC64 stream. It fuses sparse
integer HPAC evaluation, exact probability construction, and RC64 decoding.
No learned or video-derived value is embedded in the C source or binary.

The authoritative semantics oracle remains the lifted Python receiver. Native
activation is explicit (`--token-decoder native-hpac`); Python remains the
default. The retained full-field receipt proves equality for all 117,964,800
tokens, the corrected-logit and CDF-input traces, and RC64 bit position. The
native candidate is not sealed because its derived Modal total is still above
the 1,600-second fire gate.

Durable payloads and receipts live under
`/Volumes/VertigoDataTier/pact/ddm_f26q_rc64_native_20260814/`. Run
`experiments/ddm_f26q_python_reference_equivalence_test.py` to re-check the
retained evidence. The reviewed Python binding lives at
`experiments/ddm_f26q_f26_hpac_native.py`; the candidate builder copies it into
the staged runtime as `runtime/f26_hpac_native.py`.
