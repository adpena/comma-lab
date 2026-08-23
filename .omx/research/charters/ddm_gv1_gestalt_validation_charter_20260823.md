# ddm_gv1_gestalt_validation — census every measured move as {single-axis, joint} × {win, loss} and either establish the routing law or REFUTE the gestalt memo `.omx/research/ddm_gestalt_the_three_laws_20260823.md`

## MANDATE

The gestalt memo `.omx/research/ddm_gestalt_the_three_laws_20260823.md` §4 asserts a correlation from
MAIN's recall, NOT from a count:

> Every pointer move this campaign made was a JOINT move; every loss of the last two days was a
> SINGLE-AXIS perturbation.

**That assertion is unmeasured and this arm exists to try to KILL it.** If it survives a full census it
becomes a **routing law** — every future charter gets graded {single-axis → expect loss} before it
spawns, which changes what the campaign spends its arms on. If it dies, the memo's §4 and §6 are wrong
and MAIN has been pattern-matching on noise; say so plainly and the memo gets an append-only retraction.

Deliver a COUNT, not an impression.

## SCOPE

Census the dx2/rr4/cp135 lineage campaign history — every measured move that produced a byte or
distortion delta on a shipping-lineage body.

**Per move, emit a typed row:**

| field | meaning |
|---|---|
| `move_id` | arm/memo name + receipt path |
| `axes_touched` | subset of {field, model, coder, order, renderer, carrier, pose} — **derived from what the move's CODE actually mutates**, never from its prose |
| `axis_count` | `len(axes_touched)` — the classifier |
| `outcome` | `POINTER_MOVE` / `ADMITTED_SUB_BAND` / `REFUSED` / `NET_POSITIVE_COST` |
| `delta_S` | realized, from the receipt field (never a rounded display — #877) |
| `authority` | contest-CUDA / contest-CPU / advisory |
| `evidence` | receipt path + sha |

Then the 2×2 (or k×2) contingency: axis_count vs outcome, with an exact test (Fisher or equivalent) and
the effect size. **State n.** A correlation over n<10 moves is not a law and must be labelled as such.

**Seeds — routed by CONTENT and memo filename, never by bare harness id (m89: arms cannot resolve
harness ids against the repo ledger). Non-exhaustive; you must go past them:**

*Claimed JOINT wins* (each produced or was pitched as a pointer move; find each memo by glob):
- seg-edit × carrier-re-solve chain: `.omx/research/ddm_jg*` (the jg2→jg3 composition law)
- token-drop × carrier-re-solve: `.omx/research/ddm_rc4*`, `.omx/research/ddm_fs3*`
- tail-override × splice: `.omx/research/ddm_to1*` / the ma1 splice memo
- pose byte-close splice: `.omx/research/ddm_up3*`
- coupled-pair admitted micro-wins: `.omx/research/ddm_qs2*`, `.omx/research/ddm_re1*`

*Claimed SINGLE-AXIS losses* (all 08-22/23, all on the dx2 body):
- `.omx/research/ddm_ld1_lane_lossy_drop_exchange_20260822.md` (sha256 `0df6fec54e7dd2ec…`) — field
- `.omx/research/ddm_ae1*`, `.omx/research/ddm_oe1*` — model (anti-predicted, both routes)
- `.omx/research/ddm_ap1_residue_purchase_scorer_20260823.md` (sha256 `3f739cf3f71972d5…`) — residue,
  12/12 rungs
- `.omx/research/ddm_to2*`, `.omx/research/ddm_ad2*` — order (both signs)
- `.omx/research/ddm_ef1*` — estimator
- `.omx/research/ddm_ni1_247x_erratum_20260822.md` retraction section — whole-body replacement, two
  contest-CUDA authority rows

## HARD CONSTRAINTS

1. **Classify from CODE, not prose.** A memo that says "joint" while its diff mutates one array is
   single-axis. This is the #1222 genus (a structural story is not a mechanism) — read what the move
   actually changed.
2. **Confounders are the finding, not an obstacle.** At minimum test: (a) do joint moves simply spend
   MORE bytes? (b) are the wins earlier in the campaign (easier gains) and the losses later
   (harder-because-converged)? — **(b) is the most dangerous alternative explanation and would fully
   explain the pattern without any co-location mechanism.** If (b) explains it, SAY SO; that refutes
   the memo's causal claim while leaving its descriptive claim intact.
3. **Negative-existence discipline (m53, and MAIN violated it twice this week):** if you cannot find a
   move's receipt, write "did not find in `<stated scope>`" — never "does not exist," never "fabricated."
4. **No new measurement, no scorer, no Modal, no launches.** $0. This is a census over retained receipts.
5. `upstream/` READ-ONLY. Commits via `tools/subagent_commit_serializer.py` with post-edit
   `--expected-content-sha256`. `[no-triality] [p0-ledger-ok]`. No co-author trailer.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- **`ap1` (#1220, memo `.omx/research/ddm_ap1_residue_purchase_scorer_20260823.md`)** — residue purchase
  CLOSED, 12/12 registered coarsenings net-positive, 148×–81,548×, waterfill empty (0 B of the
  42,382 B demand). Its own final message says: *do not rerun current-body fixed-coder whole-group
  allocation.*
- **`ni1`/`ri1` (#1218)** — whole-body lossy re-representation CLOSED on two contest-CUDA authority rows
  (247.69× and 43.66× over their own ceilings). Token agreement understated d_seg by 349× — **token
  agreement is NOT an evaluator.**
- **`ld1` (#1212) · `ae1` (#1208) · `oe1` (#1214)** — the HPAC optimum is sharp in every direction tested.
- **`to2`/`ad2` (#1201)** — reordering is a substitute for a context model; sign depends on whether a
  context model is already present.
- **`jf1` (#1221)** — its ep2 positive control failed by 7,554 B; the memo §5 re-reads this as
  under-training. **This arm does not adjudicate jf1** — do not classify it as a win or a loss; it is
  still running.

## OPTIMAL FORM

**Reference form:** a COMPLETE census over the campaign's retained receipts with an exact contingency
test — not a sample, not a curated list, not the seed rows above.

- **SCOPE reductions (legal, declare them):** restricting to the dx2/rr4/cp135 shipping lineage;
  excluding apparatus/hygiene commits that carry no byte or distortion delta.
- **MECHANISM reductions (FORBIDDEN without an explicit TOY-BRACKET declaration that voids any family
  verdict):** classifying from memo prose instead of diffs; using the seed list as the population;
  reporting a correlation without testing confounder (b).
- **Provenance pins (sha256 prefixes, all verified this session):**
  - the memo under test `.omx/research/ddm_gestalt_the_three_laws_20260823.md` — sha256
    `7184ec1a29c43886…`, commit `97f9ea16887d`
  - residue closure `.omx/research/ddm_ap1_residue_purchase_scorer_20260823.md` — sha256
    `3f739cf3f71972d5…`
  - its final capture
    `.omx/research/arm_final_messages/ddm_ap1_residue_purchase_scorer_20260823T124528Z.md` — sha256
    `277fb754f298822f…`, commit `31a38490fb14`
  - Lane closure `.omx/research/ddm_ld1_lane_lossy_drop_exchange_20260822.md` — sha256
    `0df6fec54e7dd2ec…`
  - residue decomposition `.omx/research/ddm_ar1b_archive_residue_purchase_20260822.md` — sha256
    `388185a6c283359e…`
  - exchange rate `.omx/research/ddm_tx1_toolbox_crosswalk_20260819.md` §0 — sha256
    `4bf730e5e5d3958f…`
  - All seed rows above resolve through memo FILENAMES, never bare harness ids (m89).

## DELIVERABLE

`.omx/research/ddm_gv1_gestalt_validation_20260823.md` + a typed JSONL of the census rows, containing:

1. The full census table (every row, no truncation — if you cap, `log()` what you dropped).
2. The contingency table + exact test + effect size + **n**.
3. The confounder adjudication, especially (b) early-easy/late-hard.
4. **VERDICT: `ROUTING_LAW_ESTABLISHED` / `DESCRIPTIVE_ONLY_CAUSAL_REFUTED` / `REFUTED` / `UNDERPOWERED`**
   with `verdict_scope`.
5. If not `ROUTING_LAW_ESTABLISHED`, the exact append-only correction the gestalt memo owes.
6. `NEXT_IF_RESUMED` + `DEAD-ENDS` + `LIVE-HYPOTHESES`.

**MAIN owns all fires. This arm launches nothing.**
