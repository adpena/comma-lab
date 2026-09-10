# RLC1 receiver and driver review

Research only; score_claim=false. Two visible final passes cover the changed
receiver files plus the producer/proof scripts. Independent native review and
real-frame control are in reference_review.md. Python source tracker passes
are marked under ddm_rlc1; Ruff passes on all new source.

## Pass 1: actual data path and payload boundary

Re-read the candidate read_residual_archive path: the existing RX1 counted-tail
flag reaches RLC1.unpack_rider; version,40 weights,19 config bytes become a60-byte
config; only then does the RC64 stream reach NativeDecoder. Configlength identifies
the new mixer. Every decoded group calls observe before the next query. End-frame
geometry is checked against the actual decoded plane. Geometry never reads the
producer's complete current plane: only previous plus observed groups enter it.
The generic native runtime receives all tuning fields from the parsed rider.
Both fixed-point arithmetic and supported bounds match the independent proof.

The staging producer copies move40, changes only the7 declared paths, and verifies
header/HPAC/semantic/carrier sections unchanged. Source control reconstructs the
exact move40 RC64 stream and archive. Full field and source traces are hashed;
each trace token frame is compared with the pinned field. Native encoders have
independent contexts; all snapshots, streams, rider variants and40 container
candidates persist. Per-frame encoder+online-mixer state supports exact resume.

## Pass 2: adverse cases and independent proof boundaries

Native interval arithmetic uses signed128 products before overflow, mathematical
floor for signed division, integer square root, and nearest-even distance. The
C implementation's row window excludes the current query row; the gap mask
excludes endpoints and never shifts by64. Python adapter validates group order,
shape, alphabet and config domain before native calls. Config validation rejects
unknown versions, truncated riders, malformed bins and unsupported domains.

Reviewed fixes after they were made: Ruff's unsafe pairwise rewrite initially
removed the v[11:] slice, caught before encoding, and was corrected. Explicit C
void return signatures were added. The first public wrapper let an OS-level
PermissionError from ps escape; final wrapper catches and records UNKNOWN and
uses a fresh g2 binding. No actual decode ran under the failed wrappers. Those
files/logs are retained and do not count as proof.

The public entrypoint compiles and runs its own geometry via literal inflate.sh.
BLAS1/4 are explicit test settings; Torch remains4 threads. Neither proof imports
an alternate geometry implementation. Full raw bytes are compared against the
retained move40 raw and against its original receipt hash. Setup observers and
ordinary CUDA-gate smokes remain labeled separately. No scaler/scorer is run.

Important limit: integer geometry does not prove universal determinism of the
inherited floating mixer/HPAC/renderer. The actual two-configuration public test
is the charter's bounded determinism evidence. Native geometry is a new receiver,
so timing cannot inherit from move40 solely because the field is identical.
Whole-host concurrency is unavailable here. A valid timing seal is therefore
blocked until dwc1/MAIN supplies a matched measurement with trusted concurrency.

Assumption challenged: every uncertain scalar costs only3–6 bytes. It does not:
counting all17 geometry fields costs19 bytes. The measured candidate still clears
the30-byte admission floor, so no provenance gamble is needed to retain the win.

## Audit-bundle delegate review

Two source passes inspected python_reference_equivalence_test.py: its repository root is parents[3], and it invokes the real reference-controls main with the caller CLI unchanged. It has no local function entities. Review-tracker mark-file attempted twice, returned no ingestible entities for this .omx path; both refusals were observed, not overridden. The imported implementation has two indexed clean passes.

## Native checkpoint restart cure — two final source passes

A measured portability-of-artifacts issue surfaced during the public twins:
their state payload hashes match, but separately compiled Mach-O library hashes
do not. ReceiverCheckpoint binds library bytes, so a fresh temporary build would
refuse resume. Preserved all three exact native builds from each still-running
shell, with source hashes, exact compile-argument templates and original/retained
library facts. Both checkpoint-bound library hashes matched the retained copies.

Reviewed cached_cc.py once for source/flags/output-directory matching and once
for hash checks, refusal behavior and honest naming. It is explicitly a compiler
cache, not a compiler, and never claims to compile. It writes only the validated
requested scratch output. Actual controls reproduced all six retained libraries
and refused all six -O0 flag mutations. The runtime itself was never changed.

Reviewed resume_smoke.py once for actual saved-state input and once for its
public-shell path and completion condition. It consumes frame25 from the cold
BLAS1 run, uses the exact native cache, requests stop26, and must validate both
checkpoint binding and all26 decoded token frames against the retained source.
That concrete restart test is reported separately after execution. No resumed
suffix is admitted as cold timing or the full public identity proof.
