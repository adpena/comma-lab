# ddm_hb1 reviews

Axis: source inspection and unit tests; scorer-free; score_claim=false.

## Pass 1 — exact mechanism and consumers

Reviewed all changed code in `src/tac/decode_timing_concurrency.py`,
`src/tac/tests/test_decode_timing_concurrency.py`, and `tools/quiesced_decode_timing.py`.
Traced the pr11 JSON and nine points through classify, window adjudication, settle,
summary, assembler, calibration and the unchanged validator. Confirmed the exact v2
hash is retained. Fixed definition wrapping that inserted a space at a hyphen;
added mandatory captured-host/matched-row evidence for v3 assembly; retained settle
unwaived sums as well as counts. Rechecked those fixes and their negative controls.
The final source/test pass is tracked as hb1-final-r1; CLI pass as hb1-r1.

## Pass 2 — adversarial receipt and boundary review

Reviewed all three final changed Python files again. Checked exact comm plus integer
ppid, all four host constraints both live and captured, subvisible matching rows,
ancestor preservation, independent combined-cap violation, 25/200 activation,
100/75/175 caps, 15 samples/300 seconds inclusive, inactive-gap second bursts,
settle provisional state followed by final failure, immutable input traces, frozen
hash and timing custody, and re-derived burst/count mismatches. Validated the
calibration path reaches build_decode_wall_clock with bounded_host_baseline, while
v2 keeps daemon refusal. No new issue found on this bounded surface.

Shared assumption: ps percent CPU is a decaying average. This implements pr11's
classification envelope; it cannot prove zero interference, thermals, or bandwidth
isolation. Violating that assumption would require new instrument evidence and a
new prospective rule, not a wider exception here.

Final full test result: 89 passed, 3 failed exclusively because sandbox execution
of ps is denied. Those three tests were retained unchanged. Ruff is clean. No
live timing run or score measurement was performed. Test-fixture host simulation
is explicitly non-authoritative and is not used to freeze the real host.
