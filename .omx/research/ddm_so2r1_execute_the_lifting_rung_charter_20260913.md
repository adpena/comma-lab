# ddm_so2r1 — EXECUTE so2's one $0 rung: lossless modulo-five lifting of the move-49 field, retrained HPAC prior, real encode; verdict vs 123,998 B (charter, MAIN 2026-09-13; operator full-authority GO; Opus)

## The rung (specification is so2's memo; this charter binds the execution)
so2 (`.omx/research/ddm_so2_successor_object_priced_with_the_learned_prior_20260913.md`, sha 2c43b020…, landed bc6361908) specifies ONE
construction and ONE rung. Read its §"The one construction" and §"ONE $0 rung" IN FULL and execute them exactly: (1) materialize Z = the
reversible modulo-5 lifting of move 49's exact field F (per 64×64 patch, 2×2 cells → anchor quadrant + three (x−a) mod 5 difference
quadrants), prove the independent inverse reproduces all 117,964,800 bytes, write `Z.u8`/`Z.npz`/`cache.pt` with the trainer's cache
contract (`{'seg': uint8 tensor, 'spatial_token_sha256': …}`), pin every hash in `TRAIN_INPUTS.json`; (2) the control: the stock RLC1 pricer
reproduces move 49's live archive on F (twins); (3) train the HPAC prior on Z under cl2's law with so2's exact trainer argv (extract the
`SO2_TRAIN_COMMAND` block verbatim, then re-root every path per the STORAGE rule below and add the launcher flags it needs — MAIN measured
today that `--derive-resource-budgets` requires `--measured-peak-rss-gib` and `--measured-thread-need`; cl2's smoke measured RSS 1,697 MiB and a
15.6 GiB system-availability delta on Metal — use those as the measured inputs, or drop the derive flags and pass `--nice 0` as the day's
other launches did); terminal EMA at epoch 60 is the preregistered selection; (4) pack the epoch-60 EMA through the REAL current rail
(`_pack_terminal_ihs1` → hpr1's `prepare` body-packing sequence: rc3 encode → ck2 interleave → brotli q10/lgwin22; assert
`rc3.restore_hpac == new_ihs1`; twins; the original body through the same sequence as a member-identity control); (5) build the composition
ADAPTER so2 specifies (hpr1's altered-parts input + sj1's known-symbol injection: a new prior AND a new field through the RLC1 known-symbol
loop with the real LaneMixer/FreeCorrector/previous-plane state; two independent RC64 processes; receiver checkpoints binding Z + candidate
HPAC hashes) — read `experiments/ddm_hpr1_shape_price.py`, `experiments/ddm_sj1_rlc1_price.py`, `experiments/ddm_rq1_coarsen_and_price.py`
and reuse their producers; do not patch the stock CLI's guards; (6) encode Z under the new packed prior, twins; **J_Z = len(packed prior) +
len(stream) + 8**; verdict: PASS iff J_Z ≤ 123,998 B (the incumbent subsystem is 130,567 B: prior 11,629 + stream 118,938; so2 corrected the
charter's stale 130,525); also report J at the incumbent's own prior on Z (no retrain) as a second control, and the trained prior's
cross-entropy telemetry vs the real price (dpi1's surrogate reversal was 809 B).

## STORAGE (binding today)
Vertigo reads 38 GiB free, UNDER its 40 GiB reserve (never lower it); APDataStore reads 49 GiB free. Root EVERYTHING at
`/Volumes/APDataStore/pact/ddm_so2_first_rung/` (ExFAT: skip `._` stubs; sha-verify on the destination), set the trainer's `--min-free-bytes`
to a value APDataStore satisfies with margin (state the number), and keep the retained set ≤ 8 GiB with shas. If any stage needs Vertigo, STOP
and report the byte need rather than lowering the reserve.

## Deliverable
Memo `.omx/research/ddm_so2r1_execute_the_lifting_rung_20260913.md`: inverse-equality receipt, control row, training receipts (epochs, wall,
terminal depth histogram), pack twins, the adapter's identity controls, J_Z with its three components, both controls, cross-entropy vs price,
the verdict against 123,998 B with margins in units of the 34.8 B lottery, every boundary; serializer commits (two visible review passes per
.py; `[no-triality] [p0-ledger-ok]`; NEVER a co-author trailer or AI attribution); lane `ddm_so2r1_execute_lifting_rung_20260913` (claim it);
checkpoint `ddm_so2r1`. Heavy steps via `tools/launch_detached_process.py --output-dir <root>/<stage> --nice 0 --done-receipt …`; waits as
background receipt-only until-loops (never a foreground sleep > 3 min; never a loop that launches a successor); `sys.dont_write_bytecode`
against read-only trees. NO seal, NO Modal, NO packet: a PASS is a NEW OBJECT (the receiver must learn the SO2L inverse — a receiver change
⇒ first-measurement chain, MAIN's decision); a FAIL closes the lifting at instance/formulation scope.

## Boundaries
No Modal, no `authorize_*`, no `fire_modal_auth_eval.py`; never edit `upstream/`, the PR tree, sealed trees, contract code, receiver code, the
renderer, the carrier, or the live field (Z is a NEW artifact under your root); sj1/cb1/cr1/pp1/dpi1/rq1 directories read-only (init_depths.pt
is read from dpi1's train_inputs by sha). No ScheduleWakeup. Label MEASURED / DERIVED / INFERRED / ASSUMED.

## OPTIMAL FORM
Reference forms: cl2's law and trainer exactly as hpr1/dpi1 ran them; the real pack sequence (hpr1 prepare); the real RLC1 known-symbol rail
(sj1/rq1); so2's construction with no second variant. Declared deltas: the representation Z (the rung) and the storage root (SCOPE). Provenance
pins: HEAD (record); so2 memo sha 2c43b020…; move 49 packet/seal/leg; dpi1 init_depths.pt f05bae5b…; field sha (read by parse-back and record).

## Prior negatives accounted (operator 2026-08-15)
so1 (paper coder 1.8× wrong — this rung prices with the learned prior); dpi1 (surrogate ≠ price; the terminal EMA is packed and encoded, never
estimated); rlc/rq1 (twins, identity gate on every invocation); the day's launcher flag facts (see above); disk (Vertigo under reserve).

Final message: inverse receipt, control row, training wall, J_Z components, both controls, verdict + margins, commit shas, retained bytes,
every boundary, ending with `composition S 0.13632299781031237 @ 179,153 B [contest-CUDA T4 n600] (move 49)` unchanged (a PASS is not a score).

<!-- # FORMALIZATION_PENDING: charter, not a finding -->
