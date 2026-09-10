# RLC1 independent integer reference and native review

Owner: rlc1_review, folded into parent ddm_rlc1. No scorer or score claim.

## RECALL EVIDENCE

Read the charter/common contract, PROGRAM, operating manual, live board, and
governing NO-FAKE/native-runtime clauses. Content searches covered research
memos, CANONICAL_RESEARCH_INDEX*, sub015_DAG*, docs, and canonical_task_status.jsonl
with `fixed.point`, `lane.boundary.context`, `counted.*config`,
`receiver.*constants`, and `counted.*(threshold|geometry)`. Queried
`tools/list_canonical_equations.py --json`, filtering the JSON for
`lane_boundary_context_map_bound` and `fixed_point`.

Beyond the charter seeds, `decode_determinism_integer_arithmetic_v1` carries
the Q15 MLX cross-process experiment: one integer hash versus ten float hashes.
That separate operator does not establish this geometry's parity, but reinforces
the requirement for an independent integer oracle. The lane-boundary registry
entry explicitly excludes integer/archive bounds and successor-field transfer;
its continuous TC2 certificate cannot prove this archive price. Task ledger
ddm_no1_row3_alphabet_merge::d3b_lossless_lane_factorization records counted q8/q1
geometries that lost on their four-class quotient object; this is not a reason to
kill RLC1's conditional context map. BLP1's retained-r10 weight-floor closure is
also a different, learned-conditioner object. None of these changes the parent's
current scope; they prohibit transporting prior numbers into this cure.

## Arithmetic and overflow proof

Q16 is predeclared generic arithmetic. The 19-byte `<BHH8B6B` config counts the
class, two row boundaries, window, prior denominator, minimum count, residual
threshold, slope threshold, width denominator, width multiplier, span threshold,
and six distance edges. No provenance-incomplete tuning remains a free default.

Moments use weight `prior_den` on observed current symbols and weight one on
previous symbols. Let C be their summed weights, D=C*Syy-Sy^2,
N=C*Sxy-Sy*Sx, A=C*Sxx-Sx^2, R=D*A-N^2. All Python operations after extracting
the six moments use unbounded integers. Validity is C>=min_count*prior_den,
D>0, abs(N)<=slope_max*D, and R<=residual_max*C^2*D. Center is floor of
`(Sx*D+N*(y*C-Sy))*65536/(C*D)`. Width is the greater of
floor(65536/width_den) and isqrt of floor(width_mult*max(R,0)*65536^2/(C^2*D)).
Distances round nearest with ties to even before the counted bins are searched.

For native accepted window<=16 and prior_den<=4: C<=5120. Each slot spans
at most 63 horizontal pixels, and the history window spans at most 15 rows.
Weighted variance bounds give A<=26,011,238,400 and D<=1,474,560,000.
By Cauchy-Schwarz N^2<=A*D. Therefore R<=A*D<2^66, and even the maximum
accepted width multiplier 255 gives `R*255*65536^2 < 2^106`, safe in signed
128 bits. Raw moment sums/products before subtraction fit signed 64 bits.
Center's signed intermediate needs 128-bit arithmetic; C's truncating division
must be corrected to mathematical floor. The implementation does that.
After the residual validity gate, squared width <=255*255*65536^2<2^49,
so conversion to uint64 before integer root is safe. Admitted slope<=2 bounds
the center to the pixel mean plus at most 32 pixels of extrapolation; absolute
distance and nearest-even arithmetic are far inside signed 64-bit range.

The reference supports a wider safe config domain; native acceptance is the
shipping boundary. Neither proof says integer-Q16 bins equal historical float64
bins. The latter is a separate measured comparison and archive reprice.

## Visible review pass 1 — independent reference

Reviewed experiments/ddm_rlc1_reference.py after writing it. Verified struct
field order and that all uncertain tunables enter solely from parsed config;
the only shape/alphabet/schedule literals describe the public format. Current
moments advance only in observe after the exact expected group. Previous values
are reduced immediately; later mutation of the caller's plane cannot affect
state. Window-prefix subtraction excludes the query row. Gap and multi-run
guards independently use the decoded sentinel plane. The fitted-interval
function uses exact Python integers, floor division, and math.isqrt; no float
approximation supplies bins. Corrected Ruff's explicit strict-zip requirement.

## Visible review pass 2 — seams, signed arithmetic, control custody

Re-read the corrected complete oracle and compared equations with the parent's
native C and adapter without importing their math. Native sums before row y
contain exactly rows max(0,y-window)..y-1; adding y then removing y-window
prepares the next query. Gap mask excludes both endpoints and never shifts by
64: hi>lo+1 implies low offset+1<=62, and high offset<=63. Other-symbol bits
enter only after observed groups, matching the independent sentinel-plane test.
Native floor division, isqrt, and nearest-even match the oracle's convention.
Native config bounds make all casts and shifts safe as proved above. Adapter
validates complete groups/symbols before passing pointers to C.

Reviewed the retained run_controls.py twice: first for actual-input provenance,
independent state progression and retention-before-mismatch-failure; second for
resume binding (source/config/field hashes), atomic per-frame arrays/receipts,
fixed seeded-random sample, and failure checks. It never runs an encoder/scorer.
Ruff passes on the oracle and control runner. These are source-review passes;
parent owns review-tracker marking after indexing and final serializer hashes.

## Empirical control status

MEASURED `[macOS-CPU advisory / geometry parity only]`: final-source controls
match all 6,291,456 context bins on 32 seeded-random complete frames, zero
mismatches. Both algorithms observed all 190 HPAC groups of every selected
frame; the previous frame was the retained move40 decoded plane. Both bin
arrays per frame are retained (12,582,912 bytes), and every retained array hash
was rechecked. Sum of control-loop wall time was 7.6389100379892625 seconds;
this includes oracle work and is NOT public receiver or contest timing.
Concurrency: one control worker alongside the parent's encoder, with
OMP/OPENBLAS/MKL threads pinned to one. Launcher used --nice-best-effort and
0.1 GiB artifact budget; storage preflight found 62 GiB available on Vertigo.

Final consumer store:
`/Volumes/VertigoDataTier/pact/ddm_rlc1_rule118_cure/reference_controls_final/`.
See manifest.json (source/config/field binding and selected frame IDs),
complete.json, per-frame receipts and arrays, launch/launch_manifest.json, and
verified_summary.json. Post-run hashes match the current native C, adapter,
independent reference and repo control-runner sources. No synthetic result is
used as empirical evidence. This sample establishes measured implementation
parity on selected real frames, not all-host/all-input identity or n600 pricing.

The earlier reference_controls/ run also passed n32 but used an earlier
adapter source. Its bytes and receipts are preserved and superseded by the
final-source control. The initial SSD runner remains as its exact source;
experiments/ddm_rlc1_reference_controls.py is the landed successor with required
--resume-from and --concurrency-note options. Reviewed this successor twice
after CLI routing and Ruff cleanup, including source-drift refusal when a
different runner is pointed at an older generation. Review tracker marked both
owned Python files with two named codex passes after targeted source rescans.

## Parent driver custody review

Read experiments/ddm_rlc1_run.py without editing it. It retains all three
finished streams, twins, per-frame coder/mixer states, all container samples,
and source reconstruction; no measure-and-discard path was found in this file.
Two risks were handed to parent: INPUTS.json binds the source archive but not
the entire source runtime tree, so a separate source-tree digest must close
same-archive runtime drift before staging; the normal CPU entrypoint remains
CUDA-only except for the explicitly advisory bypass, so pr8's ordinary
contest-CPU refusal is not cured by this geometry work. The full public
entrypoint twin/timing/seal legs belong to the parent and are not measured here.

LIVE-HYPOTHESES: Native exact-integer geometry may reduce decode overhead enough
to fit the seal margin; the hot map calculation leaves Python and floating
geometry. This requires measured public-entrypoint timing. Q16 may preserve the
stream or cost only a few bytes because its spatial error is far below a pixel;
bin thresholds can still flip, so full reprice remains necessary.

DEAD-ENDS: Plain int64 evaluation of R is unsafe on valid format inputs because
the products can exceed 64 bits. Merely counting the Lane id and row band is an
incomplete compliance cure: pr8 explicitly left the other tuning provenance
unproved, so this config pays for every such operand.
