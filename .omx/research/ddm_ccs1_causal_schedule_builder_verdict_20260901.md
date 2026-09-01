# DDM CCS1 — receiver-causal schedule builder verdict

**Verdict: `CLOSED-AT-GATE` at INSTANCE scope.** CCS1 v1, seed `20260901`, with
512 nonlinear joint-context leaves produced a **664,770 B** complete archive on
the unchanged AFR1 field. The scorer-free gate is **137,986 B**. The instance is
therefore **526,784 B over** and cannot reach sub-0.12 at fixed distortion.
No scorer or authority replay was run, and the AFR1 frontier did not move.

Axis: `[macOS-CPU advisory / scorer-free exact lossless coding measurement]`.
Primary receipt:
`/Volumes/APDataStore/pact/ddm_gmf1_fitted_crossgroup_gm/ccs1_20260901/RESULT.json`,
15,362 B, SHA-256
`7ee51b8c8d6dee8fc98aeac308877e6b25df8dcc441a4120c50c2bf9e5705a43`.

## COUNTED RESULT

Every row below is a physical byte count. No entropy estimate is promoted to bytes.

| counted object | bytes | custody |
|---|---:|---|
| nonlinear model, selected XZ representation | 3,680 | `retained/model.xz.bin`, SHA `c19ccceab33edd3657bfb47fa279f4436ad0a386e93eee7626e6e9d55b6772a1` |
| serialized schedule v1 | 772 | `retained/schedule_v1.json`, SHA `07829a0757e285e8567564a4752166cdcfef99e2fe59d4d85538cdc3903fe1f3` |
| full n600 RC64 stream | 607,228 | `retained/ccs1_n600.rc64`, SHA `06571233dea915c2742c544eafa895de523a1b1b8e2e9e4f47407e2297219263` |
| replacement pool: model + schedule + stream | **611,680** | exact sum |
| unchanged semantic section | 30,856 | parsed from pinned AFR1 archive |
| unchanged carrier section | 22,010 | parsed from pinned AFR1 archive |
| residual, candidate header, and ZIP container | 224 | 96 + 28 + 100 |
| complete archive | **664,770** | `retained/archive.zip`, SHA `a56d587659864f97ce56e2a8fd5e9332ce0e36c46b1c9d651052d7976bd75fa0` |

The deterministic repeat is also 664,770 B with the same SHA. The replacement
pool is **484,754 B worse** than the shipped 126,926 B token pool and
**524,276.14 B above** the 87,403.86 B allowance. Excluding the separately
serialized 772 B schedule, model + stream is 610,908 B versus the 84,910 B gate,
an excess of 525,998 B. The complete archive is 4.8177 times the gate.

This falsifies the prior-law prediction that the causal pool would land within
plus or minus 10% of the shipped 126,926 B. The basin change was real, but this
particular compact nonlinear table discarded too much of the shipped HPAC's
predictive structure.

## LAW REGISTRATION RECEIPT

`decoder_causal_condition_transport_v1` is registered as an operational-domain
extension of
`wyner_ziv_decoder_side_information_conditional_entropy_savings_v1`, not as a
new gate. It records both obligations:

`H(E_i(C_i) | D_<i, p_i) = 0` for free conditioning, and
`B_T >= ceil(H(E(C) | D, p) / 8)` when exact context transport is required.

Registration commit:
`95f036a18c` (`canonical equations: register decoder-causal conditioning transport
[no-triality] [p0-ledger-ok]`). The registered producers include DCC1, QX3,
QX4, and GMF1; consumers include tasks #1374 and #1182 plus parser/parse-back
review. Focused canonical-equation tests passed 32/32.

## RECEIVER SCHEMA V1

The exact parser order is frame `0..599`, then global group
`g = (x mod 64) + 2*(y mod 64)` in `0..189`, then row-major sites within the
group. At each coded site the model consumes only:

- previous decoded replacement classes at `(y,x)` and `(y-1,x)`;
- current-frame classes at `(y,x-1)` and `(y-1,x)` only when their global group
  is strictly earlier;
- a 25-state boundary value recomputed from the decoded left/up/up-left/up-right
  prefix as `known_count + 5*disagree_with_previous_center`;
- deterministic tile-64 and group-bin position cells.

Frame 0 resets previous and current state to the serialized `UNK=5` value. The
nonlinear leaf key is the exact joint product of those fields; a missing leaf
backs off to the learned previous-center by group-bin row. Every CDF is five
positive `uint16` frequencies summing to 65,536. Multiplication by 32,768 gives
the RC64 total `2^31` exactly, so encoder and receiver select the same integer
CDF without floating-point ambiguity. The 772 B schedule and every 3,680 B of
selected video-derived model state are counted in the archive.

By construction, changing any current-frame class in the same or a later group
cannot change the current site's context. The unit falsifier for that condition
passes.

## FIT AND HELD-OUT CODING

The fit used LM1's temporal split: 20 contiguous 30-frame blocks; held out
blocks `1,5,9,13,17` (150 frames), leaving 450 training frames. Seed
`20260901` selected 8,192 positions per training frame, 3,686,400 sampled
training sites total. The three immutable fit checkpoints retain the sampled
positions, joint counts, and fitted raw model. The final model selected 512 of
1,876 joint contexts having at least 32 sampled observations.

Held-out coding used five independent real RC64 streams, one per held-out
30-frame block, with exact decode identity. Denominator: **150 frames,
29,491,200 symbols**. Physical streams were 63,064 + 67,628 + 60,800 + 58,816
and 89,404 = **339,712 B**, or 0.09215278 bits/symbol. A four-times extrapolation
would be 1,358,848 B and is only a held-out SCREEN, not the final byte claim.
The full n600 physical stream is the authority for this rate verdict.

The split is not a prefix. No in-sample-only claim is made. The large held-out
penalty, together with the 607,228 B full stream, shows that the selected leaf
table neither generalizes nor reproduces the shipped HPAC basin well enough.

## FULL RECEIVER CLOSURE

The candidate parsed its model, schedule, semantic section, carrier section,
residual, and stream from the exact archive bytes. Both independent full decodes
produced 117,964,800 B with SHA-256
`cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`,
byte-identical to AFR1 X.

Both independent final renders produced 3,662,409,600 B with SHA-256
`7246a4ff8f79b03ab14b3a72f6a6e2fff18b567fcb61f12a7fe311d48f5f2de7`,
byte-identical to the AFR1 custody output. The first completed render was reused
only after its full SHA check; the second was independently rendered and
selector-closed. The final successful detached launcher receipt is
`.omx/tmp/codex_runs/ccs1_20260901.done`, `rc=0`.

All payloads, including all three model coder variants, five held-out streams,
fit checkpoints, 23 full-encode checkpoints, per-frame ideal-bit ledger, full
stream, two decoded fields, two 3.66 GB renders, archive, repeat, command,
result, manifest, and done receipt remain under
`/Volumes/APDataStore/pact/ddm_gmf1_fitted_crossgroup_gm/ccs1_20260901/`.
The manifest records 46 retained files and tree SHA
`8f6b1905724b21e2b077ba34e426d182670c398ae52cc48090fadd9932218a5a`.
APDataStore had 51,784,843,264 free bytes at the first preflight and
47,810,609,152 free bytes at the successful resume preflight, both above the
mandatory 1 GiB reserve. Nothing was deleted.

## COMPARABILITY ANCHORS

The unchanged AFR1 field and archive pins passed. The three SFP1 field hashes
remain comparison anchors only and were not fitted or modified:

| field | bytes | SHA-256 |
|---|---:|---|
| B1 | 117,964,800 | `75fe37daf8c3f615cd943a76697e9c6e8eabc56cb1c23d55a6b4251fc4553690` |
| B2 | 117,964,800 | `656bd0c5c102109c3327eccd0c6e3a606aac44cbce7d9144396f8c171e24b76e` |
| B3 | 117,964,800 | `fe6a9dd8ce770e308c7c3d1903ea1e40bee44938cc836188e486eefd408f527a` |

GMF1's `RECALL_CLOSURE.json` remained SHA-256
`95f90363ea4d58b52bc00cd5370a7996dc3502b971f203d3aded6a6e71b17598`.

## RECALL EVIDENCE

The recall was not limited to the charter. Searches included content queries
for `causal`, `G/M`, `crossgroup`, `decoder`, `heldout`, `117,964,800`,
`same-schedule`, `127,606`, and `nonlinear` across `.omx/research/`, the
canonical research indexes and DAG feeds, `experiments/`, canonical equations,
and task/status stores.

Findings beyond the immediate DCC1/GMF1 seeds changed the execution as follows:

1. `ddm_jf2_terminal_diagonal_harvest_20260826.md` retained the JF1 physical
   same-schedule null row, 13,463 B model + 114,143 B stream = 127,606 B. That
   kept CCS1's claim scoped to a schedule-basin change rather than a refit.
2. `ddm_lm1_learned_model_falsifier_20260826.md` and
   `experiments/ddm_lm1_logistic_receptive_field.py` supplied the exact
   450/150 block split and the 192,118.1 B held-out linear failure. That forced a
   genuinely nonlinear joint-context model and a physical held-out encode.
3. `ddm_rxc1_restartable_exact_coder_20260901.md` and the RR2/Route-B custody
   source supplied the checkpointable real RC64 path. That changed the build
   from a screen into an end-to-end resumable physical encode.
4. `ddm_fcd1_field_for_coder_diagonal_20260829.md` and AFR1's
   `BYTE_CLOSE_REVALIDATION.json` supplied the exact field/render identity pins,
   including the full render SHA used by both final passes.
5. The broader index/DAG/task-ledger queries did not find another executable
   repaired causal G/M schema in those scopes. They did expose DCC1's rank-2
   QBT-native target-overwrite question, which stays separate from this closed
   instance.

## DENOMINATOR AND BOUNDARIES

- Tested instances: **1/1** declared CCS1 v1 instance, one seed, one 512-leaf
  nonlinear model.
- Training denominator: 450 frames; 3,686,400 fixed-seed sampled sites.
- Held-out denominator: 150 frames; 29,491,200 symbols; five physical streams.
- Full confirmation denominator: 600 frames; 117,964,800 symbols; one complete
  physical stream; two full decodes; two full renders.
- Negative scope: **INSTANCE**, not nonlinear-family nonexistence.
- Scorer runs: 0. Modal calls: 0. Authority evaluations: 0. Upstream writes: 0.
- Rate is measured. Distortion is not remeasured because the field and final
  render are byte-identical to AFR1. No score or projected score is claimed.

## DISPOSITIONS

- CCS1 v1 seed `20260901`: `CLOSED-AT-GATE` at INSTANCE scope.
- MAIN authority replay on `a56d5876…75fa0`: `FOLDED`; the mandatory byte gate
  failed, so firing a scorer would be waste and would violate the charter order.
- SFP1 B1/B2/B3 fitting: `FOLDED`; those edits remain outside this fixed-X rung.
- DCC1 rank-2 QBT-native target-overwrite grammar: `QUEUED-WITH-A-FIRE-ORDER`;
  owner = a future MAIN-assigned QX representation owner; consumer store =
  `/Volumes/APDataStore/pact/ddm_qx4/`; fire only after source proof establishes
  that the consumer wants target-overwrite output rather than historical C1
  syndrome identity. Price one new decoder-native grammar and do not rerun any
  of QX4's six closed forms.

## NEXT_IF_RESUMED

- **Disposition: `QUEUED-WITH-A-FIRE-ORDER`; owner: future MAIN-assigned QX representation owner; consumer store: `/Volumes/APDataStore/pact/ddm_qx4/`; fire trigger: source proof establishes target-overwrite consumer semantics, no duplicate QX owner is active, and the object is a new decoder-native grammar rather than any of QX4's six closed forms.** Price that one rank-2 object with a real coder and exact parse-back before any scorer request.

## LIVE-HYPOTHESES

- A QBT-native target-overwrite grammar may remove historical no-op events
  because DCC1 found 9,177 of 17,926 events do not change the decoded QBT field.
  It remains plausible only with an explicitly changed consumer contract; the
  old C1 tuple ABI cannot inherit the credit.
- A different nonlinear receiver model can still exist because the shipped HPAC
  is a causal positive control at a 126,926 B token pool. CCS1 closes only this
  sparse joint-table instance, not all nonlinear parameter-sharing models.
- Causal quotient topology remains conceptually plausible because decoded
  births, deaths, and boundaries can generate addresses without an encoder-only
  label stream, but BR2's distortion failure means topology needs a new realized
  object before rate work is justified.

## DEAD-ENDS

- Do not rerun CCS1 v1 seed `20260901` with the same 512-leaf schema: its exact
  archive is 664,770 B, 526,784 B over the gate, after full receiver closure.
- Do not promote held-out ideal bits or a four-times held-out extrapolation to
  bytes. The only final rate is the 607,228 B physical n600 stream.
- Do not use encoder-side source, target, or boundary labels. GMF1 closed those
  three SFP1 proposals at formulation scope under the decoder-causality law.
- Do not substitute fixed-G/M or position-only models for the declared three
  context families; those are weaker objects already excluded by DCC1/JBP1.
- Do not score this archive. Fixed distortion cannot overcome a complete archive
  4.8177 times the byte gate, so the scorer request is folded.
- Do not reopen B1/B2/B3 as this rate rung. Their hashes remain folded
  comparability anchors, and a lossless fit cannot repair their measured
  rendered-output harm.

Own-vehicle frontier: **S `0.14797617125559104` @ `180,002 B` `[contest-CUDA T4 n600]`, AFR1 archive SHA-256 `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25` — UNMOVED.**
