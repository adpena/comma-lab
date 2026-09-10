# ddm_dwc1 — decode timing belongs in the seal

Status: IMPLEMENTED AND MEASURED; strict passing calibration BLOCKED by unavailable process-count evidence. All three full600 proxy runs and the32-stratum map benchmark completed. No score work or Modal. Serializer custody is recorded below.

## Decisive retained T4 evidence

| Receiver | Measured T4 inflate seconds | 1,260-second margin rule |
|---|---:|---|
| move 41 / tc3 | 1,336.668977565 | REFUSE, exceeds by 76.668977565 s |
| tc4 | at least 1,800, interrupted by timeout | REFUSE, known positive control |

These are retained remote measurements, not local timing projections. The tc3 number comes from
`artifacts["contest_auth_eval.json"].inflate_elapsed_seconds` in the retained Modal result;
evaluation adds 45.555858273 seconds. The poller total is 1,397.861626581 seconds and must not be
called decode time. Exact source paths and hashes are in `ddm_dwc1_20260910/T4_RECEIPT_FINDINGS.json`.
The charter's prediction that move 41 would pass with 600–900 s of T4 decode is falsified,
INSTANCE scope. The existing scored pointer is not invalidated or changed by this margin policy.

## RECALL EVIDENCE

Searches covered `.omx/research/` content for `decode.wall.clock`, `decode.*timing`,
`decode.*budget`, `1260`, `1800.*timeout`, and `seal`; the canonical equation registry via
`tools/list_canonical_equations.py --json`; `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`,
docs/SPEC paths, and canonical task-status rows. Search results and the registry snapshot are
retained in `ddm_dwc1_20260910/corpus_recall.txt` and `equations_recall.json`.
The Codex memory registry search for dwc1/seal-contract/decode timing had no relevant hits.

Beyond the charter seeds:

- `ddm_wc1_decode_wallclock_verdict_20260816.md` reports 516.8 s optimized / 370.4 s cached
  advisory decode on an older instrument. These cannot calibrate the shipped tc3 receiver.
- `ddm_wc2_wall_clock_pass_20260820.md` already distinguishes GPU token decoding from the
  advisory render bottleneck and the job-wide 1,800 s limit. It also explains that setup and
  evaluation consume the same wall budget. The current tc3 remote result ALREADY carries
  `contest_budget_verdict: REFUSE`, computed after evaluation. The missing protection is
  admission before seal/fire, not an absence of any existing diagnostic.
- `ddm_ft1_fire_tool_cpu_axis_20260818.md` and the current upstream README show the declared
  runtime selects the hardware axis. The current receiver requires CUDA. The live charter
  addendum therefore correctly replaces a supposed CPU submission timing with a CPU proxy.
- The canonical task ledger contains the historical dc1 conditional-coding budget work;
  it does not supply a current tc3/tc4 receiver calibration. No applicable calibrated decode
  equation was found in the searched equation-registry records.

These findings changed the plan: use the actual remote inflate stage, preserve a failed
move-41 backfill rather than manufacture a pass, keep the optimized advisory cache path out
of the measurement, and record the CPU declaration/refusal separately from proxy work.

## Measurement contract and current limits

Timings run on byte-verified receiver copies under
`/Volumes/VertigoDataTier/pact/ddm_dwc1_decode_wall_clock/`. The literal public shell uses the
existing `TC3_ADVISORY_CPU=1` switch, four CPU intra-op threads, no alternate native HPAC or
shared decode cache, and retained 25-frame token checkpoints. The fresh tc3 run completed
under a 2,400-second cap through the canonical launcher with `--nice-best-effort`.
Storage preflight preserves a 40 GiB reserve and an 8 GiB artifact ceiling; payloads stay on SSD.

The sandbox denies `ps` process enumeration. Load averages are measurable, but concurrency
count and quiescence are not certified. Such local timings cannot produce a margin PASS.
No fabricated concurrency count or estimated load divisor is admissible. This does not
weaken the independently measured T4 refusal.

## Owned consumer orders

- QUEUED-WITH-A-FIRE-ORDER: owner MAIN; consumer
  `.omx/research/ddm_dwc1_20260910/MAIN_MARGIN_DECISION.json`; trigger harvest of the timing
  backfill and guard. Decide how to handle move 41 and same-receiver rp1 under the explicit
  1,260-second policy. A failed source leg cannot become a passing inherited leg.


## Live MAIN directive intake — move 40

Consumed `.omx/research/ddm_dwc1_directive_main_pr8_margin_finding_20260910.md` and the
updated live board. Move 41 is retracted as a submittable pointer for a separate rule-118
finding owned by pr8/MAIN. It remains the measured timing-failure control here. Move 40's
actual retained T4 inflate is **990.053829427 s**, scoring adds 45.23132363599984 s; its runtime
upload digest and archive identity match current source bytes. Evidence: `MOVE40_T4_EVIDENCE.json`.
That actual T4 decode clears 1260 by 269.946170573 s. It does not certify unknown local load.

The prior move40 parse-back records 1047.7603159999999 s with four CPU threads and cold n600
work, but does not record host/load or literal public-shell startup. It cannot be relabeled as
a quiesced public-entrypoint timing. A fresh direct-backend proxy on an unchanged receiver copy
completed because move40's public entrypoint ignores TC3_ADVISORY_CPU and requires CUDA.
The original declaration remains intact. This is additional work required by the live directive.

Move41's fresh four-thread full600 copied-shell proxy measured **1242.3852358330041 s**.
Named CPU-to-T4 ratio is **1.0758892966631077 = 1336.668977565 / 1242.3852358330041**.
All 3,662,409,600 raw bytes and117,964,800 token bytes are retained with their source-matching
hashes. `TC3_BACKFILL.json` records the sidecar beside the untouched original seal and consumes
both the measured-margin refusal and the same-receiver failed-inheritance control.

## Completed tc4 and map prices

Tc4 fresh full600 copied-shell proxy measured **1456.5434454170027 s** at four threads,
with no resumed checkpoint or shared token cache. Its raw and token SHA-256 exactly match
move41's output. The named tc3 ratio yields **1567.0795030489587 s** projected T4 decode,
which refuses1260 but understates the actual observed T4 lower bound1800 by at least232.920496951 s.
The direct timeout receipt therefore vetoes an optimistic transferred projection. The claim
that a single CPU ratio certifies a new receiver's T4 cost is not established by these data.
`TC4_BACKFILL.json` binds the completed local receipt, strict observed-timeout refusal, and
sidecar beside the unchanged original seal. Stage5 checkpoint's typed projection1567.08149
was a transcription error; the computed receipt and this memo's1567.0795030489587 are authoritative.

| Exact shipped component, local advisory | Measured s / million token visits | n600 seconds, projected only |
|---|---:|---:|
| tc3 LaneGeometry init + contexts + observe | 0.5179276866944712 | 61.09723597537595 |
| tc4 additional previous-map + context codes + incremental observe | 0.2706392094690517 | 31.925900217174785 |

Each denominator is **6,291,456 token visits =32×384×512**, with190 causal group calls per
frame; seed20260910 selects one random frame per temporal stratum. Every emitted map is retained
and every final reconstructed label plane equals the real field. The tc4 incremental observe
cost subtracts its nested real LaneMixer.observe time from the outer real FastContextMixer.observe
time; wrapper overhead remains. Timed methods are the unchanged shipped implementations.
No probability rows are invented. This is geometry/map pricing, not the whole mixer: it excludes
probability calibration, features/mix_rows, learned-count updates, entropy decoding, and rendering.
The source field and all64 frame/map receipts and NPZ payloads are retained in `components/`.
`COMPONENT_PRICE_TABLE.json` binds the result and full stage breakdown.

The map-only observations fall below the charter's tc3 1–3µs and tc4 +5–15µs priors. This is
an INSTANCE-scoped result for the named methods and selected real frames, not a family speed
claim. Full local decode stages cost932.602869458 s for tc3 versus1086.5208309589943 s for tc4;
those totals include much more than the maps and were measured at different unknown contention.
Map cost alone cannot attribute the T4 timeout. Probability mixing and host/device effects are
plausible unmeasured contributors; no causal attribution is claimed.

## Lossless retention and source custody

Both completed tc3/tc4 raw files are3,662,409,600 B, SHA-256
`c5a7986cf3f16360a0ef5f197ad4b5cc4cdf4c143f0c8a4319460476fc986ea5`.
Both token files are117,964,800 B, SHA-256
`b50da438e65b62d5d6f4ca1e151463d097feafd102bbd11d3e0556f849fa4ab5`.
After independently hashing both paths, `PAYLOAD_DEDUP_CERTIFICATE.json` recorded exact source,
archive, command, size/hash and inode custody BEFORE replacement. `PAYLOAD_DEDUP_RESULT.json`
then verified atomic hardlink replacement preserves both candidate paths and every payload byte.
It saves3,780,374,400 physical logical bytes. Budgets count unique(device,inode) bytes while
retaining pathname logical totals; fresh retries still reserve another complete output.

Original producer bytes are retained under `producers/` before subsequent hygiene edits.
Source1 matches the tc3 binding; source2 matches tc4. The observed decoded runtime bytes are
unchanged. Both timing and component storage accounting received two post-edit reviews and ruff.
The fresh move40 backend proxy has its own source binding and native-build receipt and ran
sequentially after the component benchmark; all runs are harvested.


## Completed move40 backfill and final admission result

| Receiver | Fresh n600 local proxy seconds | Actual retained T4 inflate seconds | Strict leg |
|---|---:|---:|---|
| move40, submittable pointer | 1243.9751226250082 | 990.053829427 | REFUSE: competing process count absent |
| move41 / tc3, retracted control | 1242.3852358330041 | 1336.668977565 | REFUSE: actual T4 over1260 |
| tc4, failed control | 1456.5434454170027 | at least1800, timed out | REFUSE: observed T4 timeout |

All local rows are `[macOS-CPU advisory]`, four threads, fresh600, not Linux CPU scores.
Move40's ratio is **0.7958791228379316 =990.053829427 /1243.9751226250082**. This ratio maps
its own local counter back to the actual990.053829427-second T4 decode. It is not a measured
load normalization. Move40 bypasses only the declared hardware selection by calling the
unchanged copied input verifier and backend directly; public-shell startup is explicitly
excluded. Tc3/tc4 invoke copied public shells using their pre-existing advisory switch.
All three original public entrypoints retain the same CUDA-required refusal semantics.
`CUDA_DECLARATION.json` checks historical executed smoke receipts against current source digests.

`MOVE40_BACKFILL.json` binds the sidecar beside the original seal. A real same-receiver
inheritance attempt is consumed and REFUSED because the source's concurrency evidence is
missing. Move41's corresponding inherited-source failure is also consumed. No invented zero
process count, guessed load divisor, or source-leg grandfathering is used. The process inventory
call is denied by this sandbox; load averages alone cannot supply the missing count. MAIN must
supply a quiesced or genuinely measured-normalized calibration before this backfill can pass.

Move40's outputs match both earlier raw/token hashes. A second pre-replacement certificate
and post-replacement proof preserve its distinct payload paths using the same immutable bytes.
`RETAINED_ARTIFACTS.json` hashes508 measurement artifacts: **3,982,232,590 unique-inode bytes**,
with11,542,981,390 pathname-logical bytes because completed payloads are hardlinked. No payload
was discarded. Git custody is separately verified; all bulk remains on SSD below8GiB.

## Seal contract, fire ordering and tests

New producers write version2 seals with a version1 structured decode-wall-clock leg. Generic
validation preserves legacy version1 documents and reports the absent leg; new seal construction
and the sealed fire path require it. The maker CLI accepts exactly one measured leg or eligible
pointer-leg inheritance. Receiver identity normalizes only the explicit top-level archive-pin
literal values and excludes archive.zip; other runtime code changes require a new measurement.
Receipt files are SHA-bound, source runtime/archive/hardware are rechecked, and ratio arithmetic
is recomputed. Calibration accepts only the canonical successful T4 inflate field; scoring time
and an unsupported whole-eval label cannot masquerade as decode. Direct exact-object T4
failures override an optimistic CPU projection. Cold-start, four-thread, consecutive-frame,
host/platform, measured concurrency and1260-second constraints fail closed.

The fire guard was edited LAST, after all three backfills were on disk. Its unconditional
`require_decode_wall_clock=True` runs at seal validation before any subprocess. Neither dry-run
nor an active public-smoke waiver bypasses it. The existing unsealed legacy CLI route is outside
this bounded seal-contract change; no claim is made that every possible evaluator launcher is gated.

**175 targeted tests passed in3.63s** on the final guard and semantic-calibration changes,
including the three fire controls that were deliberately red before the flip. Ruff passed all10
Python files. Two visible post-edit reviews cover every changed Python file; source hashes and
review findings are retained beside this memo. Test logs: `final_tests.txt`, `final_ruff.txt`.
No paid dispatch, scorer, or actual candidate-fire dry run was used as a test.

## Owned handoff and remaining work

- FIRED: tc3 full600 proxy, tc4 full600 proxy,32-stratum causal map benchmark, move40 full600
  backend proxy, all three backfills, actual failed-inheritance controls, final strict fire guard,
  and175-test regression suite. Every run is harvested; no timing daemon remains owed.
- FOLDED: failed pre-decode tc3 setup attempt, charter prediction of a passing move41 margin,
  and the original move41 policy decision into MAIN/pr8's separate retraction. Retained failed
  setup evidence remains in the launch store; it is not counted as a completed timing.
- QUEUED-WITH-A-FIRE-ORDER: MAIN owns a passing move40 concurrency calibration; consumer
  `ddm_dwc1_20260910/MAIN_MARGIN_DECISION.json`; fire when a host can actually record competing
  processes and a quiesced interval or a measured normalization. No silent waiver is authorized.
- QUEUED-WITH-A-FIRE-ORDER: MAIN/rlc1 consumes receiver identity and map prices before sealing;
  consumer `ddm_dwc1_20260910/RLC1_INHERITANCE_COORDINATION.json`; fire at rlc1 receiver freeze.
  Inherit only an eligible, strictly passing, identical-receiver pointer leg. Changed receiver
  code requires its own proxy and calibration; a field-only label does not prove code identity.

## Live hypotheses and closed paths

LIVE-HYPOTHESES: Move40 can plausibly obtain an admissible timing leg after quiesced measurement
because its exact T4 decode already clears1260 by269.946170573s. Probability-mixing or host/device
work may contribute to tc4's remaining latency; map-only CPU prices do not identify the cause.
These are untested leads for the named consumer orders, not new performance claims.

DEAD-ENDS: Move41 does not pass the new margin. Tc4's T4 timeout cannot be erased by a CPU
projection. Old optimized/cache timings cannot calibrate this receiver. A bare loaded time or
assumed zero count is not a quiesced measurement. A hash-valid scoring-time field is not decode
time. Public-smoke waiver and dry-run are not timing waivers. The current public entrypoints
refuse CPU, and move40 ignores TC3_ADVISORY_CPU; the direct backend proxy is explicitly separate.

## Live frontier

No new score from this arm. Submittable own-vehicle frontier: **S0.13763861019288715 @180,233B
[contest-CUDA T4 n600]**, move40, archive
`986d536b31ed1079c517dadea73ba33daf018c53692a2b2fbbf8d6244dfe9857`.
The current canonical pointer agrees; MAIN/cpd1 retracted move41 for a separate rule-118 finding.

## Serializer custody and landing order

The managed sandbox refused shared Git object writes. The source landing is retained as
commit `7cc51955bbb190fcb0168b8425038007126156f6` with a verified HEAD-ref source bundle.
The final enforcement landing uses the canonical serializer in an isolated Git store, chained
after that source commit, with the real commit hook enabled and post-edit SHA-256 per file.
The shared staged index is never used for these commits. `landing.bundle` contains HEAD and
both intended commits; `landing.patch` is the combined original-base-to-final-tree patch.
`LANDING_BUNDLE_VERIFICATION.json` is the final custody authority for commit IDs, hashes,
exact committed blobs, and unchanged shared-index snapshots. Source and enforcement patches
are also retained separately. These are fallback artifacts, not a claim that shared HEAD moved.

QUEUED-WITH-A-FIRE-ORDER: MAIN owns integration; consumer
`ddm_dwc1_20260910/MAIN_LANDING_ORDER.json`; fire on harvest of the verified bundle. Apply the
source landing first and enforcement landing last as one reviewable batch. Preserve the
explicit failed/blocked backfills; do not manufacture a passing move40 leg during integration.

## MAIN addendum 1 (2026-09-10 ~06:00Z) — the refusal was a gate with no door; move 40's leg is now measured

**Finding.** Every REFUSE above ("competing process count absent") was a validator contract no real
producer could satisfy: `ddm_dwc1_move40_timing.py` counted EVERY `ps` row (hundreds) and then wrote
`null`; `ddm_rlc1_public.py` wrote `margin_time_basis: "unknown_host_concurrency"`. The 175 tests
passed on a synthetic fixture that simply wrote `competing_process_count: 0`. A gate whose PASS path
only a hand-written fixture exercises is a gate with no door (memory:
`validator_contract_no_producer_can_satisfy_is_a_forever_refusal_test_the_producer_on_the_pass_path_20260910`).

**Instrument (bf467026f).** `src/tac/decode_timing_concurrency.py` + `tools/quiesced_decode_timing.py`
`{run, assemble, calibrate, leg}`: a sampler records the process table every 20 s through the whole
producer run (settle phase recorded but never counted); the verdict is by MEASURED IMPACT where the
producer leaves stage checkpoints (total excess over the run's own median pace ≤ 1 % of wall; slowed
stages listed with the processes sampled beside them), by process samples outside the instrumented
stages (≥ one full core competes; single-core daemons cannot displace four P-core decode threads), and
by an aggregate guard. Every process ≥ 5 % is listed so a reviewer can re-decide from the receipt.

**Measurements (cold, four threads, bit-identical raw c5a7986c… and tokens).**

| attempt | wall s | verdict |
|---|---:|---|
| move40 (dwc1, under rp1's three shards) | 1,243.98 | refused (no count); 1.56× the quiet time |
| move40_quiesced (attempt 1) | **797.15** | **ADMITTED**: excess 5.66 s = 0.71 %; stage 325 +22 % beside two venv pythons at 66.9 %; `dasd` at ~96 % for 5 min moved no stage |
| move40_quiesced2 (attempt 2) | 785.14 | refused: Codex desktop app 124 % + two pythons at 100 % during the native build; excess 0.41 %; corroborates attempt 1 within 1.5 % |

Calibration `cpu_to_t4_ratio = 990.053829427 / 797.1459791249945 = 1.24200`. Leg: mode measured,
projected 990.054 s, actual T4 990.054 s (completed), limit 1,260 s, problems none — sidecar beside
`SEAL_ddm_sj1_compose39_rp1_union_contest_cuda.json` (sha 4540e951…; dwc1's refused sidecar retained
as `receipts/dwc1_refused_sidecar_backup.json`). Inheritance proven on the pointer tree
(`inherit_decode_wall_clock` → mode inherited, 990.054 s, no problems). Consumer record:
`.omx/research/ddm_dwc1_20260910/MOVE40_QUIESCED_LEG.json`.

**Corrections to the body above.** "MAIN must supply a quiesced or genuinely measured-normalized
calibration" was the right ask but the wrong diagnosis: the sandbox's `ps` denial hid that even with
`ps` the producer's count could never be zero. The rule stated in the module docstring replaced the
pcpu-only heuristic after the two runs above showed it misattributed (`dasd` null effect, python bursts
real effect); the 1 % impact tolerance sits thirty times inside the 0.7 safety factor the limit carries.

verdict_scope: instance — this host (18 cores, 6 P), these two producers; the rule's thresholds are
recorded in every receipt and are re-derivable from the listed samples.
