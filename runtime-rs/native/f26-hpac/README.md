# F26 native HPAC/RC64 receiver

This directory contains the generic C lowering and audit bundle for the F26
token receiver. The reviewed Python binding is in `experiments/`. The native
function consumes archive-derived model codes,
residual values, causal plans, and the original RC64 stream. It fuses sparse
integer HPAC evaluation, exact probability construction, and RC64 decoding.
No learned or video-derived value is embedded in the C source or binary.

The authoritative semantics oracle remains the lifted Python receiver. F26R
adds direct int16 frame-context production, archive-derived int16 conv-A class
deltas, and NEON/AVX2 accumulation with a portable scalar twin. The retained
primary, repeat, and forced-scalar full-field receipts each prove equality for
all 117,964,800 tokens, corrected-logit and CDF-input traces, and RC64 bit
position. The F26R native candidate is sealed behind `native-hpac`: its measured
M5 token stage is 147.005377 seconds and its explicitly derived Modal total is
1,321.647333 seconds, below the 1,600-second fire gate.

Durable F26R payloads and receipts live under
`/Volumes/VertigoDataTier/pact/ddm_f26r_hpac_final_rung_20260814/`. Run
`experiments/ddm_f26r_python_reference_equivalence_test.py` to re-check the
retained primary/repeat/scalar evidence. The reviewed Python binding lives at
`experiments/ddm_f26q_f26_hpac_native.py`; the candidate builder copies it into
the staged runtime as `runtime/f26_hpac_native.py`.
