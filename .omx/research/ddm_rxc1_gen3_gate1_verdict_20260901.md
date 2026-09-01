# RXC1 Gate 1 — GATE-1-PARTIAL: exact suffix pricing proven, outer-loop economics unsolved

Date: 2026-09-01  
Verdict scope: **INSTANCE — RXC1 on the retained AFR1 stream**  
Axis: `[macOS-CPU advisory / scorer-free EXACT byte measurement]`  
Score claim: **none**

## Conclusion

The completed `n=32` screen lands in preregistered **Branch 1**. Both restart strides reproduce
the full exact archive delta byte-for-byte on every row: **64/64 comparisons are identical**,
with zero absolute error and 32/32 sign agreement per stride. That establishes exactness.

It does **not** pass Gate 1. The exact-vs-exact Pearson/Spearman values are
`0.9999999999999999/1.0`, which are VACUOUS-BY-CONSTRUCTION under the preregistration and carry
no gate authority. The best aggregate screen cost is **450.5928640831262 s/proposal** at stride
300: **7.5098810680521035 min**, or **50.19554561318141%** of the 897.675 s full-reencode
reference. The physical-restart-only median at that stride is still **362.9142752501648 s**
(6.048571254169413 min, `n=16`). Those costs do not enable a per-proposal SCMDL outer loop.

Terminal adaptive state also reconverges in **0/32** rows across 147 registered non-ledger
arrays. The current AFR1 instance therefore remains suffix-priced. Its routed search granularity
is **batch-of-proposals-per-suffix-re-encode**, not one suffix replay per proposal.

The raw apparatus fields `correlation_gate_pass=true`, `gate_pass=true`, and `status=PASS` in
`SCREEN.json` describe only the pre-preregistration numeric correlation gate. They are not the
campaign verdict. The binding typed verdict is **GATE-1-PARTIAL — exactness proven, economics not
solved**.

## Trigger and launch custody

The stale-precondition check passed before launch:

| AP sample | UTC | free bytes | threshold | direction |
|---:|---|---:|---:|---|
| 1 | 2026-09-01T16:13:27Z | 52,713,881,600 | 1,400,000,000 | baseline |
| 2 | 2026-09-01T16:14:27Z | 52,713,881,600 | 1,400,000,000 | stable, no decline |

The vr2 move certificate was re-read at
`/Volumes/VertigoDataTier/pact/ddm_vr2_ap_reclaim_round2/MOVE_CERT_pfs1_20260901.json` and
matched SHA-256 `6a5173ff326e667e952f5642efa06c3bfd6037e2de49a337d0b88aa8044fa7df`.

The first detached launch was fail-closed before compute: launcher counter 716 requested niceness
10, the managed sandbox denied `setpriority`, and the typed done receipt recorded rc=8 in 0.004259
s. The second launch removed the unsupported niceness claim while preserving the same command and
custody. Launcher counter 717 completed rc=0 in **7,417.677732 s** at
2026-09-01T18:20:18Z. Its retained manifest and log are under
`/Volumes/VertigoDataTier/pact/ddm_rxc1_gen3_screen_complete/monitor_attempt2/`; its done receipt is
`.omx/tmp/codex_runs/ddm_rxc1_gen3_screen_attempt2.done`.

The AP reserve was never approached. `MANIFEST.json` captured **52,434,305,024 free bytes** after
the screen and manifest stages, versus the mandatory 1,073,741,824-byte reserve.

## Resume proof — the 26 sealed rows were not recomputed

The consumed blocker receipt matched SHA-256
`581a076846dfdba0164ff5b6ab4c4818258eaa61b2591ab010e27e97885d839b`. All **26/26** blocker-pinned
`ROW.json` receipts still match their recorded byte counts and SHA-256 values. The first 26 rows
reference 51 unique underlying `RESULT.json` receipts; **0/51** had an mtime at or after the
2026-09-01T16:16:40Z successful launch. The newest was
2026-09-01T14:11:48.320000Z. Thus the expensive presealed results predate this arm and were reused.

The interrupted row resumed explicitly at frame 400 with schema
`ddm_jg4.corrector_state.v2` and all **147** state keys restored. Re-emitting the already sealed
row receipts changed no content hash; it did not re-run their exact encodes.

## Completed screen

`SCREEN.json`: 60,753 bytes, SHA-256
`e6a400be9bb140bd220b4d5e77473a36dd692e2794cbd47f48a65b30d9309420`.

| leg | denominator | stream identity | median s/proposal | mean s/proposal | median frames | Pearson | Spearman | max abs error B | sign agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full exact | 32 | authority reference | 717.6238278960809 | 739.2622890196217 | 600 | n/a | n/a | n/a | n/a |
| stride 200 aggregate | 32 | 32/32 | 477.1418477296829 | 485.89457227483217 | 400 | 0.9999999999999999 | 1.0 | 0 | 32/32 |
| stride 300 aggregate | 32 | 32/32 | **450.5928640831262** | 539.6457595847169 | 300 | 0.9999999999999999 | 1.0 | 0 | 32/32 |
| stride 200 physical restarts only | 21 | 21/21 | 476.12907699961215 | 379.99080414686443 | 400 | exact-by-construction | exact-by-construction | 0 | 21/21 |
| stride 300 physical restarts only | 16 | 16/16 | **362.9142752501648** | 382.8042958697479 | 300 | exact-by-construction | exact-by-construction | 0 | 16/16 |

The aggregate rows intentionally include early proposals whose nearest checkpoint is frame 0 and
therefore reuse that row's full exact result. The physical-restart rows isolate nonzero resume
points. Either denominator is minutes per proposal, not a local-cost outer-loop primitive. The six
new rows preserved the prior-law prediction: **12/12** new stride comparisons were identical,
raising the complete result from 52/52 to **64/64**.

Full archive deltas span the nonconstant set `{1, 2, 3, 4, 5}` bytes. The computed correlation is
therefore numerically defined, but it is still scientifically vacuous because both sides are the
same exact suffix computation.

## Manifest and retained payloads

`MANIFEST.json`: 138,824 bytes, SHA-256
`954ccada3094df4f22147d47225cab84238782291287ea51c52d1362d0a4ae2c`.

| receipt | value |
|---|---:|
| retained entries | 687 |
| retained logical bytes | 1,328,339,946 |
| free bytes after capture | 52,434,305,024 |
| sealed screen rows | 32 |
| exact full-vs-incremental comparisons | 64 |

Every new row retains its edit payload, run specifications, exact streams, terminal checkpoints,
result receipts, and row receipt. No scalar-only run, deletion, or scientific-payload move occurred.

## Binding preregistration — quoted verbatim

Source:
`.omx/research/ddm_rxc1_preregistered_harvest_adjudication_20260901.md`, SHA-256
`9d3485c46614ab888bc0c548d49a08a7420d4c2c1f6d6fc885bc3a795763223d`.

```text
1. **If SCREEN.json's incremental leg is an exact suffix re-encode** (deltas byte-identical to
   full): the correlation row is recorded as VACUOUS-BY-CONSTRUCTION and carries NO gate
   authority. Gate-1 is then adjudicated on the COST criterion alone: the measured per-proposal
   wall cost at the best stride (null-replay receipts: 716.4 s full · 475.1 s @start 200 ·
   356.0 s @start 300 · 237.7 s @start 400 — cost ∝ suffix length) vs what an SCMDL outer loop
   can afford. Expected verdict shape: GATE-1-PARTIAL — exactness proven, economics NOT solved
   (average suffix cost ≈ half of 897.675 s is no outer-loop enabler).
2. **The family's OPTIMAL FORM is named now**: restart + **state-reconvergence splice** — after
   re-encoding the edited region forward, detect when the adaptive HPAC/corrector context state
   RECONVERGES to the original trajectory; from that point the original stream suffix is reused
   byte-identically (splice), giving EXACT deltas at LOCAL cost. If the context never reconverges
   (permanently shifted count tables), that non-reconvergence is itself the decisive measured
   fact: exact incremental coding is then structurally suffix-priced on this coder, and SCMDL's
   loop granularity must be batch-of-proposals-per-suffix-re-encode, not per-proposal. EITHER
   measured outcome routes the successor; neither may be skipped by citing a vacuous 1.0.
3. **If the incremental leg is halo-approximate** (deltas differ from full): the chartered ≥0.9
   Pearson/Spearman gate applies exactly as written, plus max-abs-error and sign-agreement.
4. **No GATE-1-PASSED headline may be written at harvest without quoting this rule** and stating
   which branch (1/2/3) the receipts land in.
```

### Computed adjudication

- Branch fired: **1 — exact suffix re-encode**.
- Exactness: **proven for 64/64 comparisons** on this `n=32` AFR1 instance.
- Correlation authority: **none; VACUOUS-BY-CONSTRUCTION**.
- Best aggregate cost: **450.5928640831262 s/proposal at stride 300**.
- Cost versus the 897.675 s reference: **0.5019554561318141×**, a 1.9922086467716347× speedup,
  but still 7.51 minutes for one proposal.
- Gate verdict: **GATE-1-PARTIAL**.

## State-reconvergence result

At terminal frame 600, **0/32** full-edit rows exactly match the baseline adaptive state across the
147 registered non-ledger arrays. Per row, **57–72** arrays differ, median **69.5**. The final
`previous` plane is equal in **32/32** rows.

Because the post-edit suffix is otherwise identical and the state transition is deterministic,
exact reconvergence at an earlier point would keep the subsequent state trajectory identical.
Terminal mismatch therefore rules out an earlier stable exact reconvergence in these 32 rows.
This closes splice-on-reconvergence for the measured AFR1 instance; it does not claim global
nonexistence for other coders, reset schedules, or representations.

## Consequence for XOV1 and JC1

The sg2b receipt, SHA-256
`ed592b68c663df80247f3f6a14103f93931c4c7a45b38966da9863a449b4a39a`, binds the SCMDL X-alone
axis closed: all 3/3 realized X-only legs were refused. Gate 2 is therefore **G/M-coupled only**;
no standalone X field should be repriced.

The three XOV1 alternatives from SHA-256
`ad093da3358996cb30700a2d0976af2436b10ea570eeada276580336b6ce6345` remain finite candidates,
not an outer-loop search space:

1. **Born context/expert.** The 47,603-byte generator packet is counted, leaving at most 37,306 B
   for every other `G+M` byte and requiring replacement of at least 89,620 B of the current
   126,926-byte pool. It needs an exact joint batch price; no additive estimate can admit it.
2. **Generator-conditioned peel chain.** The `Lane→MyCar→Undrivable→Movable` chain conditioned on
   generated GF1 class and boundary distance must be priced as a joint G/M alternative. Its entropy
   estimate has no authority.
3. **5,506-record directed B/H/W support.** The support is a new rate-first candidate but remains
   selected by a token-GT proxy. It gets an exact joint price only; a smaller receiver-closed archive
   would still owe fresh pose compensation and a matched n600 Seg/Pose gate.

These alternatives are non-additive unless a union is explicitly built and jointly priced. RXC1's
result permits a bounded **batched** exact pricing pass over the three alternatives; it does not
permit one independent suffix re-encode per proposal inside a general SCMDL loop.

JC1's live requirement remains a 39,522.14-byte cut from the 126,926-byte joint field/model pool to
reach the 87,403.86-byte allowance and the 140,479.86-byte affine archive ceiling. This result leaves
the joint G/M mechanism live but keeps its per-proposal refit loop computationally non-fireable.
JC1 must first batch proposals behind one suffix replay, or change the counted causal-state design
to a measured bounded-reset form that restores local exact pricing. No scorer lane may fire merely
from this coder-side partial gate.

## RECALL EVIDENCE

The bounded recall pass searched `.omx/research/`, arm-final messages,
`CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, design/spec surfaces, current hot state, canonical task
status, executable coder code, and the canonical-equations registry. Queries included `rxc1`,
`restartable exact coder`, `state reconverg`, `splice`, `suffix`, `SCMDL`, `XOV1`, `HPAC`, and
`RC64`.

Recall found no existing AFR1 post-edit state-reconvergence or splice receipt in those scopes. It
did find the canonical `token_rate_model_direction_dependence_v1` and
`greedy_set_average_vs_marginal_price_v1` laws, which reinforce exact physical pricing and forbid
transferring an average byte price to a joint candidate. That changed the harvest by retaining the
terminal-state comparison and routing the result to batching; it did not authorize a splice build,
scorer run, or coder redesign in this arm.

## Verification and boundaries

- Focused apparatus tests: `16 passed in 0.32s`.
- Screen and manifest stages: rc=0.
- No scorer, CUDA, Modal, network dispatch, archive evaluation, or pointer write ran.
- `upstream/` remained read-only.
- No sg2b or XOV1 store was modified.
- The current work measured exact bytes and CPU wall time only; it did not lower a score.

## NEXT_IF_RESUMED

- **Disposition:** `QUEUED-BATCH-PRICE`; **owner:** task #1374 SCMDL G/M owner assigned by MAIN; **consumer store:** `/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder/`; **fire trigger:** the three XOV1 source hashes still match and a retained batch runner can price all three alternatives per suffix replay without discarding any candidate payload; **action:** price born expert, generator-conditioned peel chain, and 5,506-record support as separate joint G/M rows in one bounded batched pass.
- **Disposition:** `QUEUED-STRUCTURAL-LOCALITY`; **owner:** task #1374 exact-coder builder assigned by MAIN; **consumer store:** `/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder/`; **fire trigger:** before any general JC1 per-proposal outer loop is launched; **action:** implement and screen either batch-of-proposals-per-suffix pricing or a counted bounded-reset causal-state form, with exact identity and retained payloads as the admission gate.
- **Disposition:** `QUEUED-CONDITIONAL-AUTHORITY`; **owner:** MAIN scorer/evaluation scheduler; **consumer store:** `/Volumes/APDataStore/pact/ddm_jc1/scmdl_projection/` plus the canonical candidate/evaluation ledgers; **fire trigger:** a receiver-closed repeat-identical archive is at most 140,479.86 B and has fresh full-n600 Seg/Pose component receipts satisfying the exact score law; **action:** request the normal sequential contest-authority evaluation only then.

## LIVE-HYPOTHESES

- Batching the three finite XOV1 alternatives may amortize one causal suffix traversal across multiple G/M prices; it is plausible because the measured cost is dominated by deterministic suffix replay, while the alternatives are known before the replay starts.
- A periodic or bounded reset of adaptive HPAC/corrector state may recover exact local pricing; it is plausible because the measured nonreconvergence is carried by persistent adaptive state, but it changes the counted causal model and must earn its bytes.
- Born geometry may help as a decoder-derivable context without the full 47,603-byte generator packet; it is plausible because the disagreements are structured by lane/horizon geometry, but no such derivation was found in the searched receiver code.
- The 5,506 directed support may contain a smaller joint rate-and-distortion subset; it is plausible because it is a new cross-parent support, but weak because the same token-GT selection principle failed realized transfer on FCD2/FCD3.

## DEAD-ENDS

- Calling the raw `SCREEN.json` correlation or `gate_pass=true` a Gate-1 pass is closed: exact-vs-exact correlation is vacuous under the binding preregistration.
- Per-proposal suffix re-encoding as a general SCMDL outer loop is closed on this instance: the best aggregate median is 450.593 s/proposal and terminal state reconverged in 0/32 rows.
- Splice-on-reconvergence for these AFR1 rows is closed: 57–72 adaptive arrays still differ at frame 600 on every row.
- Reopening X-alone field edits is closed on the current body/instrument: sg2b refused all 3/3 doses and Gate 2 is G/M-coupled only.
- Treating the three XOV1 alternatives as additive, admitting them by entropy estimates, or using B/H/W labels as scorer evidence is closed by their own custody memo.
- Reopening the exact XOV/QX residual carrier is closed: its 661,618-byte pool is 534,692 B worse than the shipped 126,926-byte pool.

**OWN-VEHICLE FRONTIER: UNMOVED — S = 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600], AFR1 archive SHA-256 `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`.**
