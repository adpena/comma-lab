# TC3 move40 rebind: independent two-pass source review

Reviewer: ddm_tc3 geometry subarm. No trace, encode, fit, receiver, native build
or scorer was launched by this review. No source, runtime, pointer, shared
index or canonical state was edited. This note is the only reviewer write.
Parent owns executable controls, full measurements, source indexing, review
tracker and serialization. Parent reports Ruff green for all three files.

Final reviewed SHA-256:
- `experiments/ddm_tc3_move40_trace.py`:
  `1b3fe73a64cde250bc409af937939a504517a2b124d40593ba1c3c2ec980b0b1`.
- `experiments/ddm_tc3_move40_price.py`:
  `08d10b13029545d5aadbbec68cb9a32c5655081f05a8347ad1f5caee58552868`.
- `experiments/ddm_tc3_move40_receiver.py`:
  `a0d403ffb71a3fbfa1abe2169787ef547753c4317ca497f46de11e0c82b71f34`.

## RECALL EVIDENCE

Used the earlier complete TC3 charter/common-contract recall and source
reviews, especially `native_trace_state_review.md`, `mixer_price_review.md`,
`receiver_review.md` and `recovery_review.md`. Read the new files and their
complete diffs against the frozen move39 files. Searched the new source for
`move39|ddm_tc3_trace|ddm_tc3_price|ddm_tc3_receiver|180186|8877f75d|4aa519a2|ddm_rp1`.
The remaining old receiver import is deliberately only a generic stable-build
helper; its helper sources are now separately bound.

Read the actual move40 source runtime and compared its native wrapper, native
C and checkpoint source hashes with the earlier native-state review. All
three match exactly. The source shell and residual reader are identical to
the reviewed source; the source inflate.py differs only in archive SHA/size.
Read the owned raw-copy manifest. This changed the review focus to complete
native state, transitive helper custody and removal of the external-raw-path
dependency. No old-field gain or receiver result transfers to move40.

Current rebind archive is 180,233 B,
`986d536b31ed1079c517dadea73ba33daf018c53692a2b2fbbf8d6244dfe9857`.
The new field is 117,964,800 B,
`b50da438e65b62d5d6f4ca1e151463d097feafd102bbd11d3e0556f849fa4ab5`.
The nested owned root is
`/Volumes/VertigoDataTier/pact/ddm_tc3_lane_predictor_seal/rebase_move40/`,
with execution stages under `move40/` and baseline custody under `retained/`.

## Corrections before final passes

The initial trace called `NativeFreeCorrector` without its required library
argument. Parent fixed it to use the library selected by the stable native
build helper. Parent also bound the stable helper's build/identity and storage
source modules, and added a closed-handle restore check plus exact shape,
int64 dtype and value-one validation for the saved nonempty-frame `have_prev`.
Receiver now binds both the owned baseline receipt and copy manifest before
staging/proof. All findings were reported before the final clean passes.

## Trace visible pass 1: mechanism and rebinding — CLEAN

Read the complete trace and final changes. The current archive and field pins
feed a new immutable input manifest and distinct trace store. The native
constructor enforces its compiled configuration and receives the selected
library path; ABI1, 23-family source layout and plane geometry are checked.
The source-bound native build is separate from the RC64 encoder library.

The trace still runs the actual HPAC group plan, residual correction, native
corrector row, unchanged TC1 mixer and real RC64 twins against the current
field. It observes decoded labels in the original order. Native coding rows
are float32 as required by the existing mixer and coder. No probabilities or
state are borrowed from the old field. The only old-arm helper used is the
generic stable build path; it does not call the old arm's pin or prepare.

Shared assumption: this frozen native corrector reproduces the current public
source trajectory for the same labels. Existing public use and matching code
support it; the current full shipped-stream reconstruction remains the gate.
No unresolved source blocker.

## Trace visible pass 2: complete checkpoint and restart — CLEAN

Re-read the frame lifecycle and final snapshot wiring against the complete C
state analysis. The 79 arrays are copied, and `have_prev` is stored separately.
Capture occurs only after all groups, corrector.end_frame, mixer.end_frame and
the updated previous plane. Restore uses a fresh native handle, checks the
exact key set/dtypes/shapes and restores have_prev. It does not serialize raw
pointers, depend on scratch values or reuse an in-flight Python group token.

Both encoder snapshots, mixer snapshot, completed frame number and previous
plane accompany the corrector state. State and frame artifacts remain
immutable, source-bound and retained. Native C/library, ABI accessor source,
stable-build helper, build/identity helper and storage helper are bound. A
resume must match the current trace binding and snapshot bytes. Full trace
still refuses unless the two actual envelopes agree with the current shipped
envelope. No unresolved source blocker; fresh-process implementation controls
and full n600 parity are parent-owned, unmeasured by this review.

## Price visible pass 1: current-field consumers and fixed features — CLEAN

Read the complete price file. Its only changes from the reviewed frozen price
file are the trace import and move40 root. Preparation therefore consumes
current native-trace frames and captured current trace binding; per-frame
lineage and field checks remain intact. Both zero-extra reconstruction checks
remain active. Fixed-feature A/B calibration, causal observe order and counted
coefficient layout are unchanged.

Shared assumption: correct native trace rows can replace correct Python trace
rows as offline inputs without altering the feature algebra. That is conditional
on the trace controls, not a claim that a small field change has a small rate
effect. No unresolved source blocker.

## Price visible pass 2: arithmetic, retention and scope — CLEAN

Rechecked the full stage bindings, subset objective, iteration snapshots,
integer conversion, all-symbol codelength sums and twin serialization against
the previously reviewed implementation. There are no formula or precision
changes. Full current-field pricing remains required for both variants; no
move39 result is loaded as a gain or optimum. The scope string still limits
the convergence gap to the retained fitting subset and excludes universal
geometry/full-population optimality claims. Every actual twin payload and
stage checkpoint remains retained. No unresolved source blocker.

## Receiver visible pass 1: owned runtime and archive path — CLEAN

Read the current receiver and source-runtime patch targets. The rebind uses
the current archive SHA and 180,233 B guard. The fixed container menu, source
archive reconstruction, all n600 twins requirement, B-versus-A 300 B selection
rule, minimum 30 B net gate, counted 40 coefficients and exact changed-file
census remain unchanged. The actual current source's shell and residual reader
still match each required replacement exactly by source inspection.

Shipping modules retain relative imports and online geometry. Candidate and
frontier structured smoke, explicit advisory CPU mode and unchanged default
CUDA gate remain required. No unresolved source blocker. The assumption that
the current encoded stream decodes to the same field is tested by the actual
current candidate's public proof, not inherited from move39.

## Receiver visible pass 2: baseline custody and recovery — CLEAN

Re-read binding, stage recovery, public resume and final identity joins. The
baseline raw and parse-back receipt are now owned copies. Their copied paths
need not equal the source receipt's original path; the full comparison checks
bytes/SHA and source archive identity. The tiny receipt and COPY manifest facts
are bound before staging/proof, and the full raw is hashed when proof completes.
This removes the earlier dependency on another arm retaining an external raw
path while preserving its provenance.

The COPY manifest records source and owned raw as 3,662,409,600 B with SHA
`c5a7986cf3f16360a0ef5f197ad4b5cc4cdf4c143f0c8a4319460476fc986ea5`,
and source archive SHA matching move40. This reviewer read the manifest; parent
performed the full copy/hash check. Native build selection, frame checkpoints,
recovery attempt directories and completed-receipt checks remain intact.
No unresolved source blocker; no current-field public identity, cross-host
parity, contest runtime or score is claimed here.

Both passes apply to the three hashes above. Parent may proceed to its
authorized short trace/resume controls, then full current-field work if those
pass. Any additional Python source edit resets the applicable review passes.

Disposition: FOLDED into parent ddm_tc3's active move40 rebind. Owner parent
ddm_tc3; consumer the nested rebase_move40 store; fire trigger these completed
clean source passes and the parent's short real-input implementation controls.

LIVE-HYPOTHESES: the native trace may shorten current-field rebinding while
preserving the exact source stream because its complete learned state and
public arithmetic are retained. New A/B gains and public identity remain
execution questions; the old-field results are historical.

DEAD-ENDS: the missing-library constructor cannot launch and has been corrected.
Reusing move39 fitted gains, adaptive state or public proof as move40 evidence
is invalid. Depending on another arm's movable baseline raw path is unnecessary
once the verified owned copy supplies the same bytes.
