# ddm_pk3 — pk2's own named reopening: the scorer-free PK2R preflight that gates a QAT run.

**Owner:** codex arm · scorer-FREE (that is the point) · no Modal · no training launch in this arm

## OPTIMAL FORM (read first)

Reference form: a counted, scorer-free byte+MSE projection over the real CPR1 pose carrier, sufficient
to decide whether a rate-aware gauge + QAT run is worth firing. Declared reductions: SCOPE only —
projection may use the banked carrier arrays rather than a re-fit; it must still be the FULL basis and
coefficient population, not a slice. MECHANISM reductions are TOY-BRACKET: a projection that does not
run the real Huffman/Rice coders, or that estimates MSE analytically instead of measuring the
carrier product, cannot satisfy the trigger.

**Authority:** full online research, full OSS, full internal leverage (our code/docs/unwired modules
are off the shelf — adapt, refactor, extend, fork). PR130 repro code is off-the-shelf authorized.
Cite path + commit for anything reused.

## WHY THIS EXISTS

`ddm_pk2` (cfddfc503a) closed the pose-carrier REPRESENTATION axis at INSTANCE scope: 135 candidates,
49 scored at n120, and **CPR1 stayed best on every row**. Its own NEXT_IF_RESUMED left exactly one
live reopening with an explicit trigger:

> `ddm_pk2_rate_aware_gauge_qat`; owner = future PR130 pose-carrier training owner; fire_trigger =
> **a scorer-free counted PK2R preflight projects at least 2,000 full-archive bytes saved with
> carrier-product MSE below 2.5e-6**, after which only a resumable QAT run and a seeded stratified
> n120 row may fire.

**Nobody has run that preflight.** It is scorer-free, it is cheap, and it is the gate on the only
un-refuted pose-rate mechanism.

## THE CARRIER, READ AT SOURCE (verify before relying on it)

`/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/code/carrier_codec.py`
(READ-ONLY intake — copy out to work):

- `HEADER = struct.Struct("<4sII")` at **:14** — magic + basis_bit_count + coefficient_bit_count
- `encode_compact_carrier` at **:289**, docstring: *"Encode exact 5-bit basis codes and 12-bit
  delta/zigzag coefficients"* — the basis is ALREADY 5-bit, do not propose descending to it
- basis path: `_zigzag_signed(basis_codes, BASIS_BITS)` → `_encode_huffman` (**:308-309**), canonical
  Huffman with computed code lengths at **:54/:128**
- coefficient path: `_encode_rice` at **:208** with per-column optimal `k` search at **:222**
- per-dimension `basis_scales` / `coefficient_scales` as float32 (**:301-305**)

Measured section total: **23,384 B = 12.24% of the archive = 0.0155704 S marginal.**

**What is already REFUTED — do not re-run it:** generic-basis substitution. pk2 measured
`basis_dct_keep75` (−24 B, d_pose 0.00225570165), `keep50` (+64 B, 0.0555), `keep25` (+464 B, 0.580)
and the `basis_dct_packet_k{064,128,256}` family, against a control d_pose of **2.0148458e-05**. Every
one destroys pose by orders of magnitude. The fitted basis is genuinely video-derived and genuinely
needed at fidelity. Low-rank coefficient arms (`coeff_lowrank_*`) also lose. Read
`.omx/research/ddm_pk2_pose_carrier_representation_20260809/RESULTS.md` before proposing anything.

## YOUR JOB

Run the preflight pk2 specified, and report the trigger as MET or NOT MET.

1. **Project the byte saving** of a rate-aware gauge (the coding-cost-aware choice of basis/coefficient
   gauge, as opposed to the current fidelity-first fit) through the REAL Huffman + Rice coders on the
   real carrier arrays. Not an entropy estimate — the actual coders, the actual bit counts.
2. **Measure carrier-product MSE** for each projected gauge. The trigger bar is **< 2.5e-6**; the
   shipped control's realized d_pose is 2.0148458e-05, so state the relationship you are assuming
   between carrier-product MSE and realized d_pose, and label it DERIVED or MEASURED — do not let an
   assumed relationship carry the verdict silently.
3. **Report the trigger**: ≥2,000 B saved AND MSE < 2.5e-6 → MET, and hand the QAT successor a sealed,
   resumable ticket (do NOT launch it). Otherwise NOT MET → the pose axis is closed on this vehicle at
   this scope, and you say so plainly.
4. If the projection lands **near** the bar, say how near and what would move it. A 1,900 B projection
   is not a pass and must not be reported as one.

## HARD RULES

- `upstream/` IMMUTABLE. Intake clone READ-ONLY — copy out to work, never `git add` inside.
- NO training launch, NO scorer run, NO Modal in this arm. It is a gate, not the thing it gates.
- Bulk artifacts → `/Volumes/VertigoDataTier/pact/ddm_pk3_20260809/`. No `/tmp` in evidence.
- Commits via `tools/subagent_commit_serializer.py`, POST-EDIT `--expected-content-sha256`, tags
  `[no-triality] [p0-ledger-ok]`, **no attribution trailer of any kind**.
- `.py`: 2 × `tools/review_tracker.py mark-file <f> --status reviewed`; never `REVIEW_GATE_OVERRIDE=1`
  with a `.py`. Always `.venv/bin/python`. `score_claim=false` throughout.
- `ddm_cx2` (composition) and `ddm_tm1` (token model) are LIVE. Do not touch their stores or scope.

## DELIVERABLE

The projected byte saving and measured carrier-product MSE per gauge candidate, through real coders,
with the trigger stated MET or NOT MET and the sealed QAT ticket only if MET.
