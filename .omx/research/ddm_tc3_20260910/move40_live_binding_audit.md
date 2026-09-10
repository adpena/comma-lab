# TC3 move40 live binding audit

Independent bounded read-only audit of the current source bindings and retained
receipts during the move40 trace/prepare run. No measurement, native build,
scorer, runtime mutation, state/index change or commit was launched. The
checkpoint inspected below is the immutable stage 0340 snapshot, not a claim
about the eventual n600 result.

Reviewed source SHA-256 values remain identical to the two-pass move40 review:

| File | SHA-256 |
| --- | --- |
| `experiments/ddm_tc3_move40_trace.py` | `1b3fe73a64cde250bc409af937939a504517a2b124d40593ba1c3c2ec980b0b1` |
| `experiments/ddm_tc3_move40_price.py` | `08d10b13029545d5aadbbec68cb9a32c5655081f05a8347ad1f5caee58552868` |
| `experiments/ddm_tc3_move40_receiver.py` | `a0d403ffb71a3fbfa1abe2169787ef547753c4317ca497f46de11e0c82b71f34` |

## Verified custody

The current owned store is
`/Volumes/VertigoDataTier/pact/ddm_tc3_lane_predictor_seal/rebase_move40/move40`.
Its 47 manifested source files and complete retained field were rehashed and
matched `INPUTS.json`. The archive is
`986d536b31ed1079c517dadea73ba33daf018c53692a2b2fbbf8d6244dfe9857`,
180,233 bytes. The field is
`b50da438e65b62d5d6f4ca1e151463d097feafd102bbd11d3e0556f849fa4ab5`.
No move39 archive SHA, field SHA or 180,186-byte literal remains in the three
move40 source files. The external directory name `ddm_sj1_compose39_price`
contains the current move40 source; the directory name does not identify the
field version.

The baseline copy is under
`rebase_move40/retained/source_parseback/`. `COPY.json` joins the owned raw to
the original raw and source archive in retained `PARSEBACK_RESULT.json`:
3,662,409,600 bytes, raw SHA
`c5a7986cf3f16360a0ef5f197ad4b5cc4cdf4c143f0c8a4319460476fc986ea5`.
The retained parse-back report names the current archive, pair_count 600 and
current decoded-token SHA. This audit checked the owned raw's size and the
receipt joins; it did not repeat the producer's full 3.66 GB raw hash. The
public receiver independently hashes both full raw files at proof completion.

| Retained baseline metadata | SHA-256 |
| --- | --- |
| `COPY.json` | `a2a3bd6dd274ca50da525c8273fa2ad6e21abd80159f8e5dcb9e8e9db85e77d4` |
| `PARSEBACK_RESULT.json` | `82e16070b62d7aed08ba13f921d9112405b6a308254c6847e37ce25d689a929c` |

The receiver binds both metadata files before staging/public work. Its final
comparison reads the owned raw and retained source receipt; it does not reopen
the external original raw path. Candidate source/runtime/field validation also
uses the owned manifest. The existing completed-proof fast path rehashes its
retained output artifacts after the candidate/source binding guard.

## Native libraries and checkpoint dependencies

The immutable trace stage 0340 checkpoint rehashes to
`4eae112818be04c6d9e86aaea079002fcd6551600802b2a8d127568acbfe9352`.
Its binding equals the trace binding captured by `prepare_v2/INPUTS.json`.
Both selected native source and library facts were rehashed and matched
`trace/corrector_native/SELECTED.json` and the trace checkpoint binding. The
selected directory is owned:
`trace/corrector_native/attempts/attempt_0000`. Both C sources are in owned
`source_runtime/`; neither native library is loaded from a move39 payload tree.

The import of `ddm_tc3_receiver` in the move40 trace uses its generic
`stable_libraries` helper. That call reaches `proof.identity`,
`proof.build_libraries`, and storage/record helpers; it does not call the old
receiver's move39 pin, stage, prepare or public functions. The old receiver,
its native-build helper and its storage helper are explicit file facts in the
trace binding and still rehash correctly. Their source files must remain
unchanged while these frozen trace receipts are reused.

The snapshot is taken after all groups and both frame-end updates. Its native
state includes the copied persistent arrays and `have_prev`; restore validates
array census, shape and dtype, closed group state, ABI 1 and 23-family layout.
The full binding also covers the retained selected native libraries and the
native checkpoint source. This audit supplements the prior state-layout and
two-pass source reviews; it does not replace the full n600 source-control check.

## Remaining operational dependency

The pipeline has not detached every stage from the external source tree.
`ddm_tc3_move40_trace.prepare()` lines 95–106 and 117 rehashes external `LIVE`
and `FIELD`, scans `LIVE`, and verifies/copies them even when owned `INPUTS.json`
already exists. `ddm_tc3_move40_price.binding()` line 59 calls that function for
each pricing stage. Trace resumes and future pricing invocations therefore
still require the external current runtime and token checkpoint to remain
available and unchanged. Their disappearance would fail closed; it would not
silently switch to a different field. The receiver itself does not call
`prepare()` and uses the owned source/field/raw/receipt after those are pinned.
Parent acknowledged this dependency and kept the bound source unchanged.

## Handoff correction

The in-progress main memo still had a move39 “PENDING: recovery receipt
completion” sentence, although its opening correctly stated that the recovery
guard refused the stale field. Parent acknowledged that this action is
CLOSED/SUPERSEDED by the move40 proof, and that historical move39 table rows
will lose their stale “current” label when the memo is rewritten. The historical
memo is separately retained. No new review-generated queue row is required.

## RECALL EVIDENCE

This continuation used the completed `move40_review.md`,
`native_trace_state_review.md`, `recovery_review.md` and
`measured_comparison_review.md` reviews, the current charter/common contract,
live move40 input/native/checkpoint/baseline receipts and the main memo.
Bounded source queries were `move39|8877f75d|4aa519a2|180186|ddm_rp1|ddm_tc3_trace|ddm_tc3_receiver|LIVE|source_parseback|stable_libraries|native_`
and `binding|ref.prepare|source_frame|inherited|ROOT|LIVE|FIELD|resume`.
They identified the real external source availability dependency and the stale
historical recovery action; no additional algorithm or measured result was
introduced. Earlier full-corpus mechanism recall remains in the geometry and
main-arm recall records.

LIVE-HYPOTHESES: none introduced by this custody audit.

DEAD-ENDS: move39 public-receipt recovery as a current-field completion path is
closed by the pointer guard and superseded by the owned move40 proof. The old
generic helper import is not evidence of a move39 payload dependency: its
executed build/storage path consumes owned move40 source and retained libraries.
