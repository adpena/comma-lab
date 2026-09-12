# SO1 — keep the renderer's pre-image, change how its fine cells are described

Owner: `ddm_so1`. Charter: `charters/ddm_so1_successor_object_with_preimage_freedom_charter_20260912.md`.
Contract: `../tmp/codex_runs/_common_contract.md`. Design complete; first measurement **QUEUED-WITH-A-FIRE-ORDER**.
`research_only=true`, `score_claim=false`, `promotion_eligible=false`, `pointer_moved=false`.

**No construction in the receipts proves both halves.** The best next question is whether a coarse
copy of the **current pre-image field**, plus a small counted table of fine-cell patterns and exact
overrides, fits in **94,292 B**. This construction preserves every token's freedom: an override may
name any address and any of the five labels. It never asks a smooth generator to carry the renderer's
bias without paying for the exceptions. Its untested assumption is the total **coarse + table + exact
residual** price, not that correct labels automatically render correctly.

The requested generator/sparse-residual and Lane proposals already have substantial adverse evidence
beyond the charter. Those precedents materially change the ranking. The new table construction is a
candidate description, not an implemented submission or a claim of a new mathematical method.

## Authority and accounting

Number labels throughout: **M** = MEASURED in the cited receipt, with its axis; **D** = DERIVED here;
**A** = ASSUMED for a stress calculation, not a prediction; **I** = INFERRED mechanism. Identifiers,
dates, dimensions, code literals, and definition constants are not empirical measurements. A range
formed from A inputs is a **conditional scenario interval**, never a confidence interval. The
machine-readable arithmetic is `ddm_so1_20260912/arithmetic.json`; source hashes are in
`ddm_so1_20260912/source_pins.json` and `input_pins.json`.

M, independently rehashed here: move-48 archive, **179,111 B**, SHA-256
`d830edd371641e1968765ae6be27120a6c55a9b1ca4b3747158e603a43ef149c`, at
`/Volumes/VertigoDataTier/pact/ddm_hpr1/public/retrain_frame_even/candidate_runtime/archive.zip`.
Its ZIP contains one STORED member `p`, **179,011 B**. Parsing the actual RX1M header gives
`(RX1M, 1, 2, 0, 250, 11629, 29862, 18450)`.

| Current component | B | Evidence / interpretation |
|---|---:|---|
| Arithmetic token stream | M 118,896 | TMX1 control48, exact n600 coder / local CPU advisory |
| HPAC member | M 11,629 | Actual archive header |
| Semantic renderer member | M 29,862 | Actual archive header; REN2 byte-identity control |
| Carrier member | M 18,450 | Actual archive header |
| Everything else | D 274 | Archive minus preceding four entries; includes ZIP, RX1M, tail riders/prefix |
| Whole archive | M 179,111 | Actual stat and SHA above |

The charter's **18,624 B remainder** is D carrier + tail overhead excluding the **100 B ZIP**:
`18,450 + 174 = 18,624`. Its four listed masses sum to **179,011**, not **179,111**.
This memo keeps the missing ZIP explicitly. In every primary candidate budget below, the complete
non-stream remainder **D 60,215 B** is retained, including the HPAC member even when unused.
Deleting the unused prior could buy **D 11,629 B** before changed framing; that credit is **not taken**.
This gives the charter's approximately 95 KB representation gate a precise, conservative meaning.

M inherited authority: `[contest-CUDA T4 n600]`, `d_seg=0.00010345`, `d_pose=0.00000459`, from the
move-48 receipt mirrored at
`experiments/results/modal_auth_eval_mirror/contest_auth_eval_ddm_hpr1_first_measurement_t4_run1_20260911.json`.
No new scorer ran. Recompute using `upstream/evaluate.py:63-65,92`:

```text
D r = 25 / 37,545,489 = 6.658589531221714e-7 S/B
D D48 = 100*0.00010345 + sqrt(10*0.00000459) = 0.01711995387438173
D S48 = D48 + r*179111 = 0.13638261682704697
D Bmax(real) = (0.12-D48)/r = 154507.26560515573 B
D strict integer cap = 154507 B; demand = 24604 integer B
D replacement gate = 154507 - 60215 = 94292 B
D S(154507) = 0.11999982314442906; S(154508) = 0.12000048900338219
```

These use the stored component precision that reproduces the pointer. They do not infer a CPU score
or unrounded hidden scorer outputs. All candidate S values below are D conditional on exact field
reconstruction, identical renderer/carrier consumption, and a compliant runtime. They are not scores.

The target is **the coded field**, not GT: M **117,964,800 B**, all **600 × 384 × 512** cells,
`/Volumes/VertigoDataTier/pact/ddm_sj1_pass6/retained/fields/subset6.u8`, SHA-256
`a92e7d902a4498961217f02c2b90d3fb9025901ba6d047201ff3bf297fa2f7a8` (rehashed here).
The move-48 provenance already binds that field; TMX1 independently reproduced it. `pass6.u8` is a
different, unshipped field. **No read or write of `ddm_sj1_pass7` was made.**

<!-- # FORMALIZATION_PENDING: this is a design of a counted coarse-stencil table plus exact overrides, not a measured law. Exact-output substitution follows determinism and upstream/evaluate.py:92; the existing generator_form_fit_error_entanglement_v1 and decoder_causal_condition_transport_v1 equations prohibit a free fit credit or unavailable decoder state. The new description has no empirical anchor until the queued full-field coder rung runs. -->

## RECALL EVIDENCE

The common contract's historical frontier paragraph is superseded for this arm by the live pointer,
hot-state POINTER_LINE, charter, and GS3 Addenda **47–53**, all read at source. AGENTS.md and
CLAUDE.md currently have identical SHA-256 `62a27f9e69144c2c9d38de539d0a953e0641d82de209c0fda942ef6ed7dfb71f`.
PROGRAM.md and the craft handoff were read before edits. The active claims identify another arm's
live pass 7, which this arm does not consume or duplicate. This arm owns no scorer lane.

Search receipts in `ddm_so1_20260912/` preserve the following scopes and queries:

* `content_recall_paths.txt`: research Markdown/JSON/JSONL, design docs, canonical task ledger and
  P0 ledger; `pre.?image|sparse residual|coarse.*lattice|boundary regulariz|syndrome|tropical.*SDF|generator.form`.
* `focused_recall.txt`: content across `.omx/research/ddm_*.md`;
  `coarse.{0,40}(token|lattice)|boundary regulariz|lane.{0,50}(generator|8.dim|spline)|syndrome|permutation.*(token|label)|residual.*(pre.distort|edit)`.
* `index_dag_tasks_recall.txt`: `CANONICAL_RESEARCH_INDEX*`, every `sub015_DAG_*`, and
  `.omx/state/canonical_task_status.jsonl`; `pre.?image|generator|residual|boundary|syndrome`.
* `equations_recall.json`: successful `.venv/bin/python tools/list_canonical_equations.py --json`;
  relevant subset in `equations_relevant.json`. Source and argparse inspection covered GF1, HG1,
  GDC3, LTG1, D3C, and the checkpoint/serializer/launch tools. The small memory-registry search
  returned no SO1/pre-image/common-contract hits; no memory-derived numerical fact is used.

What changed the plan **beyond the seeds**:

1. **GF1 and its capacity decomposition**: the much-cited **M 2.178×** generator-form advantage is
   inseparable from approximately **M 1.12%** fitting error. GF1 measured **M 47,603 B packet +
   385,448 B exact residual = 433,051 B** on its older field, with **M 1,325,033 mismatches**.
   `generator_form_fit_error_entanglement_v1` explicitly retracts a transferable lossless credit.
   The dominant gap was bulk boundaries, **M 67.43%**, not Lane alone. Therefore A cannot inherit
   a 48 KB exact backbone simply because HG1 once made a 48 KB lossy one.
2. **BZ2D and OBX1** separate two born objects the charter juxtaposes: BZ2's own measured n600
   distortion is **M 5.266919383676105**, on a local PyAV advisory axis. QBT2B r10's physical
   **M 106,714 B** archive has **M 8.626700750101595** distortion on the later n600 local CPU
   instrument. Its **M 0.327712** was the old n32 HT estimate, not BZ2's distortion and not n600.
   No rank here combines BZ2's byte count with QBT2B's n32 distortion. XO1 MAIN §7–9 corrects its
   own mistaken “no receiver” reading; this memo consumes the corrected packed-section identity.
3. **D3B, D3C, LTG1, BLP1** already priced the Lane split and its topology. D3B's real old-field
   quotient is **M 49,696 B** and conditional Lane packet **M 64,276 B**; the full subsystem
   **M 127,499 B** loses to its matched joint coder. LTG1's exact topology/shape object costs
   **M 233,262 B**, and BLP1's inherited predictor weights alone cost **M 60,191 B**. These prevent
   treating “8 dimensions” or “Lane is one third of bits” as an achieved Lane representation.
4. **GDC2/GDC3 and LS1's audit of GDC4**: correction rates are generator- and geometry-local.
   Long boundary runs measured **M 0.246234 / 0.494987 B per mismatch** on two different fields;
   GDC2's re-price of the scanline teacher was **M 0.685364 B/mismatch**. GDC4's adaptive and
   dense-floor numbers are entropy/counting-model costs, **not compressed payloads or universal
   lower bounds**. I retain its endpoint accuracy counterexample but not the stronger claim
   that every possible learned run generator was ruled out by a scalar cost.
5. **SE3 / RL1 / SP1** already hold cheaper descriptions than naive point indices, but on different
   objects: SE3 **M 81,365 B** requires a receiver-derived support chart; RL1's **M 14,553 B** is
   an evenly strided n32 mask crop; SP1's n600 **M 421,366 B** support incumbent beats its contour
   code. None is a free/current-field residual price. This directs the rung to real coded bytes.
6. **MXO1 followed to the CURRENT timing leg**, rather than inheriting either old slack figure:
   move-48 `.decode_wall_clock.json` records **M 1,023.261602487 s** and policy limit
   **M 1,260 s**, leaving **D 236.738397513 s** for incremental receiver work. The raw
   **D 776.738397513 s** before the 1,800-second limit ignores scorer/runner work. The charter's
   approximately 700 seconds is not the admissible incremental budget. This closes no new
   algorithm on timing; it requires a new measured receiver leg if the byte rung passes.
7. **FB1's old single-axis statement has expired.** At move 48, setting renderer bytes to zero
   while holding distortion gives **D S=0.11649873676891269**. Thus “no single axis perfected
   reaches 0.12” is false on the current arithmetic. It remains a valid dx2-object receipt;
   REN2/RQ1, not FB1's old numbers, supply the current empirical renderer negatives.

The cross, SJ1, REN1/REN2/RQ1, DPI1/TMX1, and born-object memos supplied the seed evidence; their
later corrections were consumed. MD1's **M 62.011%** persistent error belongs to QBF1's n32
EMA trajectory, not all 600 current planes or all successors. MC1's motion-conditioned coding
negative is a declared predictor screen. BD1's temporal saturation is a declared counting-model
screen, not a universal theorem. The new spatial coarse-stencil construction adds decoded future
spatial context without importing their temporal savings. No existing n600 exact-byte result for
this particular counted table and its overrides was found in the searched sources.

## The conserved object and the price that matters

Let F be the shipped pre-image field, C a counted coarse description, and G(C) a deterministic
prediction. Store exact exceptions E at **every** location where G(C) differs from F. The receiver
computes `F_hat = overwrite(G(C), E)`. Admission requires `SHA(F_hat)=SHA(F)` over the full field.

This is a real mechanism for retaining per-token pre-image freedom: any token may be overwritten,
including isolated non-GT labels. It is not a direct inverse solver. Compression may choose a
different description of F, while the existing renderer and carrier see F itself. Deterministic
same-input rendering implies unchanged Seg/Pose, subject to receiver implementation and runtime
validation; the memo does not treat that implication as a new exact score.

The controlling inequality is **D `K(C,G) + B(E) + H <= 94,292`**, where H counts the actual
envelope and every video-derived table. Generic fixed algorithms are free; learned tables are
counted. No SegNet/PoseNet weights, evaluator-at-runtime call, GT image, or free GT table is needed.
All existing renderer bytes are a **banked PR130 third-party artifact** (REN1), not original SO1
work. HG1 and the coders are existing Pact machinery; the proposed coarse-pattern grammar is the
only SO1 design contribution, with no claim of literature novelty.

## A — generator plus the pre-distortion edits

If a generator produced the true partition exactly, only its intentional renderer-bias edits
would remain. But for an actual approximate generator A,
`F - G_A = (GT - G_A) + (F - GT)` in an overwrite/composition sense, not ordinary label subtraction.
The supports can overlap or cancel; summing pass edit counts is not an exact final residual.

| Pass / field | M edited tokens | M marginal bits/edited token | Receipt and limitation |
|---|---:|---:|---|
| Pass 1 footprint | 9,179 | not established for this exact footprint in the read receipts | SJ1 §1; its cited 4.718 is MODELLED, not a charge |
| Pass 2a | 7,804 | 6.2307 | SJ1 §§5–6, +6,078 real stream B |
| Pass 3 | 1,339 | 5.3055 | SJ1 §§10–11, +888 real stream B |
| Pass 4 full field | 415 | 6.4000 | SJ1 §19; shipping subset is smaller |
| Pass 5 full field | 235 | 7.7957 | SJ1 §26, +229 real stream B; only a subset shipped |
| Pass 6 full / shipped subset | 489 / 335 | 5.2537 on the subset only | SJ1 §34, subset +220 stream B |

The gross pass count **D 19,461** is neither unique addresses nor the current edit set. REN2 §6
reports **M 18,900** current-field differences from DALI GT; intervening RP edits and subsets
prevent deriving that total by adding these rows. The pass-1 price is deliberately left unknown;
JG4's **M 3.8217** belongs to another **M 10,900-token** set and does not fill that cell.

Paper price requested by the charter: applying the observed late-pass range **A 5.2537–7.7957
bits/token** to the **M 18,900** final edits gives **D 12,411.9–18,417.3 B**. This is a transferred
*marginal dense-stream* model, **not** entropy or a coded standalone sparse residual. In particular
it does not pay an independent address stream. An explicit generic enumerative code can instead
name an m-element support out of N positions plus one of four replacement labels per site:

`D ceil((log2 binomial(117964800,18900) + 18900*2)/8) + 64 = 37,982 B`.

This is a constructive paper upper budget for a fixed-weight support plus values and a 64-byte
header, not a minimum entropy, a retained payload, or a timed decoder. Using **A 47,779 B** for an
**exact** GT generator gives **D 120,406–145,976 B whole archive**, **D S 0.097293–0.114319**.
The interval brackets the optimistic transferred marginal bill and the explicit support code;
it is conditional on a generator that **has not been found**. It demonstrates that pre-image
freedom itself is affordable, not that HG1 satisfies the premise.

Actual HG1 evidence: GF1's **M 433,051 B** replacement on an older lineage would cost **D 493,266 B**
with today's conservative fixed remainder, or **D S≈0.345566** if it closed today's F. That is a
cross-object scale calculation, not a remeasurement. The current pre-distortion footprint is two
orders of magnitude smaller than HG1's fitting gap. **Do not re-run unmodified HG1.**

**Single falsifier:** the proposed generator's real n600 complete `packet + exact residual + envelope`
exceeds **D 94,292 B**. For HG1 this is already a FORMULATION negative on its measured old fields;
for a new generator the count/geometry-to-byte law must be remeasured, not transferred.

## B — coarse tokens plus a deterministic boundary regularizer and residual

Keep one anchor in each 2×2 block: `C=F[:,::2,::2]`. Decode all C before fine details, so both
left and right coarse neighbors are available. A fully specified first regularizer is a fixed
local categorical vote: predict the upper-right cell from `(C,E)`, lower-left from `(C,S)`, and
lower-right from `(C,E,S)`, with lowest-label ties and clamped frame edges. Anchors stay fixed.
This is one deterministic local regularization step, not a claimed global Potts solver. It
encourages short straight boundaries; arbitrary exact residuals restore dashes, corners, and bias.
No learned state or scorer runs in this decoder.

Price: `D B_B = 60,215 + C_bytes + R_B + H_B`. Coarse dimensions are **D 600×192×256**.
The frequently tempting `C_bytes=118896/4=29,724` is **A**, not a measured scaling law: thin
boundaries can keep most of the entropy at half resolution. Use it only for a transparent stress
point. Also assume **A 88,304** residual cells, the *old GDC1 K=8 teacher's measured count used as
a target scale*, not evidence about this regularizer; **A H=512 B** is a design reserve. Applying
the two different measured residual rates **A 0.246234–0.685364 B/cell** gives:

* **D whole B 112,194–150,971**, **D conditional S 0.091826–0.117646**.
* At the expensive end, the residual must contain at most **D 93,462 cells** for these assumed
  coarse/envelope costs. This is a screening threshold only; the real coder decides.
* Doubling the assumed coarse bill to **A 59,448 B** adds **D 0.019792 S**, taking the expensive
  endpoint above target. The uncertainty can therefore reverse the apparent feasibility.

These are not statistical bounds. With no measured C or residual on F, a narrower actual byte
interval would be invented. The incumbent remains the available fallback at **M S 0.1363826**.
The first rung prices this regularizer as the control for D below.

**Single falsifier:** its exact n600 `C + residual + actual envelope > 94,292 B`.
Failure closes this fixed anchor/vote/codec INSTANCE, not multiresolution coding or every regularizer.

## C — an eight-parameter Lane generator plus the rest coded as today

LS1 measured **M 691,677 Lane symbols / 117,964,800**, **M 320,719.636 Lane bits /
957,986.402 total bits = D 33.4785%** on the same field under the *older move-44 coder*.
Applying that share to today's stream grants **A 39,804.6 B** of removable Lane cost. With all
other charges fixed, Lane generator plus exact residual could then cost at most **D 15,200.9 B**.
An explicit **A 8 parameters/frame at 16 bits** costs **D 9,600 B** before framing and leaves
only **D 5,600.9 B** for residuals. Hundreds of bytes from temporal prediction is unmeasured here.
Dash births, widths, gaps, occlusion and rare geometry still need to be described.

Even the 39,804.6 B grant is not a separable-code theorem. The true chain is
`L(F)=L(quotient)+L(Lane occupancy | quotient)`. Surprise assigned only to true Lane symbols omits
non-Lane occupancy confirmations; changing the factorization changes CDFs everywhere. D3B already
measured that difference: **M 49,696 B quotient + 64,276 B Lane packet**, plus model/framing,
reproduces its exact older field and does **not** beat its joint stream. The actual Lane conditional
was much larger than the class-attributed surprise. The old quotient is not today's measured price.

The atlas-grant scenario gives **D B 148,906** and **D S≈0.116271** only at **A zero Lane residual**;
at **A R=5,600.9 B** it merely reaches the **D 0.12** boundary before new framing. That interval
is not a prediction of an eight-parameter fit. LTG1's *actual* exact trunk+residual grid was
**M 239,818–382,611 B**, and its best direct shape was **M 221,717 B**; neither supports a small
residual for the same finite grammar. BLP1 closes the retained learned Lane head at its weight floor.

**Single falsifier:** recode the complete n600 quotient-plus-Lane-generator-plus-residual object and
find the whole replacement **>94,292 B**. Only an explicitly new positional/shape mechanism reopens
the old LTG1/D3B forms. The eight-parameter hypothesis itself remains unmeasured, with a weak prior.

## D — a counted coarse-stencil transducer for the pre-image's fine phases

Use B's anchors, but encode the fine pattern *conditioned on its decoded coarse neighborhood*.
A stencil is `(center, north, south, west, east)`, each a five-class label: **D 5^5=3,125 keys**.
For each key store three modal labels for the three missing positions in the 2×2 block, computed
from the **current F**, not GT. There are **D 9,375 labels**, packable at three bits each into
**D 3,516 raw B**. This is a finite, fully counted dictionary, not hidden learned code. The table
can reproduce a recurring jagged or intentionally wrong fine pattern as easily as a smooth one.
Every remaining cell is an exact override. Fitting the table is integer histogramming, with stable
ties; no gradient training or model launch is proposed by this memo.

The algorithm is old-fashioned dictionary prediction; the new design choice here is which object
and conditional it codes. It differs from a temporal context overlay, an endpoint generator, and
the born RGB correction lattice. The decoded coarse field supplies spatial lookahead that the
original group-causal coder cannot use directly. It cannot create information: it wins only if the
complete description represents that information more cheaply than the shipped factorization.

Per key and child, the mode minimizes Hamming errors among constant-label predictors, so
**D M_D <= M_B** for B's deterministic key-dependent votes. That does **not** imply fewer coded
bytes: overrides can become more scattered, and the table itself costs bytes. The honest ranking
criterion is **D `R_B - R_D > actual_table_cost + envelope_difference`**. This is why B and D are
measured together in **one** first rung.

At the same **A C=29,724**, **A M_D=88,304**, **A H=512**, and the same transferred residual-rate
stress band as B: **D B_D=115,710–154,487**, **D S=0.094167–0.119987**. The upper endpoint has
only **D 19.6 B** of real-valued headroom; it is **not robust** and misses a separate 30.04-byte
margin. At the high residual rate the count screen is **D M_D<=88,332**, but an exact payload
price, not that count, is the fire predicate. The table needs to remove roughly **D 5,131–14,280**
equally priced residual cells to repay its uncompressed size at the two transferred rates; actual
geometry may change this. If the coarse field costs twice the assumed amount, this design also fails
its expensive endpoint. No measured current-field entropy interval is claimed.

**Single falsifier:** the best complete exact n600 B/D packet exceeds **D 94,292 B**. More specifically,
D loses the rank to B whenever its table buys fewer real coded bytes than it costs. A failure is an
INSTANCE negative for this five-point/2×2/table/codec grammar. Enlarging the stencil is not an
automatic next action: the key table grows exponentially and must first get a new paper budget.

## Ranked result — feasibility scenarios, not invented score measurements

All rows target exact reconstruction of F and hence **D D48=0.01711995387438173**, conditional on
receiver closure. All intervals depend on the assumptions stated above; the data do not license a
strict expected-S ordering of the overlapping live rows.

| Priority / conditional S order | Construction | D whole-archive B scenario | D conditional S interval | Evidence disposition |
|---|---|---:|---:|---|
| 1, unresolved B/D pair | B: coarse + fixed regularizer + exact residual | 112,194–150,971 | 0.091826–0.117646 | Unmeasured C and residual; control in the one rung |
| 1, unresolved B/D pair | D: same coarse + counted fine-pattern table + exact residual | 115,710–154,487 | 0.094167–0.119987 | Best new mechanism to test; wins only by its real net table saving |
| Counterfactual, not dispatch rank | A: **exact** 47,779 B GT generator + pre-distortion | 120,406–145,976 | 0.097293–0.114319 | Exact generator absent; available HG1 exact old-field replacement is 433,051 B |
| 2, weak live hypothesis only | C: eight-parameter Lane, granting atlas savings | 148,906–154,507 before new framing | about 0.116271–0.12000 | Requires residual 0–5,601 B and a transferred class-bit credit; prior exact forms fail |

Thus the answer is **a coarse-conditioned fine-pattern description with unrestricted exact escapes**,
if the specified byte measurement passes. There is no proof today that it passes. A small token
error, a low entropy estimate, or an oracle cell count alone would not discharge the missing bill.

## The ONE $0 first rung — exact bytes and full-field decode, no scorer

Owner at fire: **MAIN's SO1 byte-successor**. Consumer store:
`/Volumes/VertigoDataTier/pact/ddm_so1_first_rung/RESULT.json` and this memo's B/D rank.
Disposition: **QUEUED-WITH-A-FIRE-ORDER**. Trigger: MAIN harvests this design, claims a local byte-only
lane, confirms a new owned output root and storage/resource admission, and revalidates the field,
source-tool and pointer pins. No dependency on pass 7; if F changes, invalidate the budgets and
rebind the rung before starting. This charter expressly forbids launches, so the rung was **not run**.

Expected quantity: a physically retained B/D packet price, not an entropy. The paper stress point
is **A C≈29,724 B** and **A M≈88,304**, yielding the scenario intervals above. Confidence is low;
GF1/GDC2 repeatedly missed transferred residual projections. **Falsifier:** neither B nor D produces
an exact field with packet **<=94,292 B**. A packet **<=94,261 B** reserves at least **D 30 B**
inside the target; byte-pass alone queues a receiver build/timing check, never an evaluator launch.
A real result above the gate closes the concrete grammar even if its mismatch count looks good.

The following is an exact command **sketch**, against read existing APIs, not a new installed CLI.
It enumerates the full population, retains all coder candidates/repeats, and independently decodes
the packet before it reports a passing byte result. HG1's `encode_residual`, `coder_race`,
`apply_residual`, and `et1.compress_payload/decompress_payload` were read at source. In particular
`build_packet` was **not** reused: it hardcodes HG1 stream IDs and would mislabel this new grammar.
Its miniature SO1P envelope is research-only and not accepted by the shipped inflate runtime.

Execute the marked Python block after extracting it to the byte-successor's owned source file;
the command spelling for its declared resume argument is:

```bash
sed -n '/^# SO1_FIRST_RUNG_BEGIN$/,/^# SO1_FIRST_RUNG_END$/p' \
  .omx/research/ddm_so1_successor_object_with_preimage_freedom_20260912.md \
  | .venv/bin/python - --resume-from /Volumes/VertigoDataTier/pact/ddm_so1_first_rung
```

For an actual detached launch MAIN must use `tools/launch_detached_process.py` (verified argparse:
`--output-dir`, `--cwd`, `--purpose`, `--authority`, `--derive-resource-budgets`, `--walltime-cap-s`,
`--done-receipt`, `--arm-watchers`, `--dry-run`, then `--` command). The pipe above specifies the
measurement's content; it does not bypass that launch discipline. No launch command was executed.

```python
# SO1_FIRST_RUNG_BEGIN
import argparse, hashlib, io, json, os, shutil, struct, sys
from pathlib import Path
import numpy as np
for directory in (Path.cwd()/'src', Path.cwd()/'experiments'):
    sys.path.insert(0,str(directory))
from experiments import ddm_hg1_heterogeneous_analytic_generator_gate as h

def sha_file(p):
    digest = hashlib.sha256()
    with Path(p).open('rb') as f:
        while chunk := f.read(1 << 20): digest.update(chunk)
    return digest.hexdigest()
ap = argparse.ArgumentParser()
ap.add_argument('--resume-from', type=Path, required=True)
root = ap.parse_args().resume_from.resolve()
allowed = Path('/Volumes/VertigoDataTier/pact/ddm_so1_first_rung')
assert root == allowed and not allowed.is_symlink()
root.mkdir(parents=True, exist_ok=True)
assert shutil.disk_usage(root).free >= 42 * 1024**3
src = Path('/Volumes/VertigoDataTier/pact/ddm_sj1_pass6/retained/fields/subset6.u8')
expected = 'a92e7d902a4498961217f02c2b90d3fb9025901ba6d047201ff3bf297fa2f7a8'
assert sha_file(src) == expected and src.stat().st_size == 117964800
pointer = json.loads(Path('.omx/state/canonical_frontier_pointer.json').read_text())
assert pointer['effective_frontier']['archive_sha256'] == 'd830edd371641e1968765ae6be27120a6c55a9b1ca4b3747158e603a43ef149c'
pin_rows = json.loads(Path('.omx/research/ddm_so1_20260912/source_pins.json').read_text())
tool_paths = {str(Path(h.__file__).relative_to(Path.cwd())), str(Path(h.et1.__file__).relative_to(Path.cwd()))}
bound = {row['path']:row for row in pin_rows if row['path'] in tool_paths}
assert set(bound)==tool_paths
for row in bound.values(): assert sha_file(row['path']) == row['sha256']
F = np.memmap(src, mode='r', dtype=np.uint8, shape=(600,384,512))
def keep(name, data):
    p = root / name
    if p.exists():
        assert p.read_bytes() == data, ('resume divergence', str(p))
    else:
        h.et1.atomic_bytes(p, data)
    return p
memo = Path('.omx/research/ddm_so1_successor_object_with_preimage_freedom_20260912.md')
keep('INPUT.json', json.dumps({'field_sha256': expected, 'schema':'so1-stencil-v1',
    'design_sha256':sha_file(memo),'hg1_sha256':sha_file(h.__file__),
    'coder_sha256':sha_file(h.et1.__file__)}, sort_keys=True).encode())
def race(name, data):
    raw = keep(name+'.raw', data)
    out = h.coder_race(name, raw, root)
    win = out['winner']; fact = out['coders'][win]['coded']
    return win, Path(fact['path']).read_bytes(), data
def key(c):
    q = np.pad(c.astype(np.int32), 1, mode='edge')
    nswe = (q[:-2,1:-1], q[2:,1:-1], q[1:-1,:-2], q[1:-1,2:])
    k = c.astype(np.int32)
    for v in nswe: k = 5*k + v
    return k, nswe
def predict(c, table, kind):
    k, (_,s,_,e) = key(c)
    if kind == 'D': vals = table[k]
    else:
        vote = np.stack([(c==v).astype(np.int16)+(s==v)+(e==v) for v in range(5)],axis=-1)
        vals = np.stack((np.minimum(c,e),np.minimum(c,s),vote.argmax(-1)),axis=-1)
    f = np.empty((384,512),dtype=np.uint8); f[::2,::2] = c
    f[::2,1::2]=vals[...,0]; f[1::2,::2]=vals[...,1]; f[1::2,1::2]=vals[...,2]
    return f
C = np.ascontiguousarray(F[:,::2,::2])
coarse = race('coarse', C.tobytes())
if len(coarse[1]) > 94292:
    result={'disposition':'INSTANCE_REFUSED_COARSE_BYTES_ALONE','coarse_bytes':len(coarse[1]),
        'gate_bytes':94292,'field_sha256':expected,'score_claim':False,
        'axis':'macOS-CPU scorer-free exact coarse bytes','fine_stages_run':False}
    facts=[h.et1.file_fact(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ('MANIFEST.json','RESULT.json')]
    h.et1.atomic_bytes(root/'MANIFEST.json',json.dumps(facts,indent=2).encode())
    h.et1.atomic_bytes(root/'RESULT.json',json.dumps(result,indent=2).encode())
    print(json.dumps(result,indent=2));raise SystemExit(0)
counts = np.zeros((3125,3,5),dtype=np.int64); start = 0
for stop in (120,240,360,480,600):
    p = root / ('counts_%03d.npy' % stop)
    receipt = root / (p.name+'.json')
    if p.exists() and receipt.exists():
        fact=json.loads(receipt.read_text());assert fact['sha256']==sha_file(p) and fact['field_sha256']==expected
        counts = np.load(p, allow_pickle=False); assert counts.shape == (3125,3,5)
        start = stop; continue
    for t in range(start,stop):
        k,_ = key(C[t]); ys=(F[t,::2,1::2],F[t,1::2,::2],F[t,1::2,1::2])
        for j,y in enumerate(ys):
            counts[:,j,:] += np.bincount((5*k+y).ravel(),minlength=15625).reshape(3125,5)
    buf=io.BytesIO(); np.save(buf,counts,allow_pickle=False); keep(p.name,buf.getvalue())
    keep(receipt.name,json.dumps({'sha256':sha_file(p),'field_sha256':expected},sort_keys=True).encode());start=stop
table=counts.argmax(-1).astype(np.uint8)
bits=((table.ravel()[:,None] >> np.arange(3)) & 1).astype(np.uint8)
packed=np.packbits(bits.ravel(),bitorder='little').tobytes(); assert len(packed)==3516
table_row = race('table',packed)
def envelope(rows,kind):
    body=bytearray(struct.pack('<4sBB',b'SO1P',1,int(kind=='D')))
    for coder,coded,raw in rows:
        body.extend(struct.pack('<BII32s',h.et1.CODER_IDS[coder],len(raw),len(coded),hashlib.sha256(raw).digest()))
        body.extend(coded)
    return bytes(body)
def unpack(blob):
    magic,version,kind=struct.unpack_from('<4sBB',blob); assert (magic,version)==(b'SO1P',1)
    cur=6; raws=[]
    for _ in range(3):
        cid,n,m,sha=struct.unpack_from('<BII32s',blob,cur); cur+=41
        raw=h.et1.decompress_payload(blob[cur:cur+m],h.et1.CODER_NAMES[cid]);cur+=m
        assert len(raw)==n and hashlib.sha256(raw).digest()==sha;raws.append(raw)
    assert cur==len(blob)
    return kind,raws
results=[]
for kind in ('B','D'):
    pred_path=root/(kind+'_prediction.u8')
    if not pred_path.exists():
        tmp=pred_path.with_suffix('.partial')
        with tmp.open('wb') as f:
            for t in range(600): f.write(predict(C[t],table,kind).tobytes())
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,pred_path)
    pred=np.memmap(pred_path,mode='r',dtype=np.uint8,shape=F.shape)
    for t in range(600): assert np.array_equal(pred[t],predict(C[t],table,kind))
    tr=table_row if kind=='D' else race('empty_table',b'')
    for order in ('frame_raster','tile16_time'):
        raw=root/(kind+'_'+order+'.residual.raw')
        if not raw.exists(): h.encode_residual(F,pred,raw,None,order)
        rr=race(kind+'_'+order+'_residual',raw.read_bytes())
        blob=envelope((coarse,tr,rr),kind); packet=keep(kind+'_'+order+'.so1p',blob)
        keep(kind+'_'+order+'.repeat.so1p',envelope((coarse,tr,rr),kind))
        tag,raws=unpack(packet.read_bytes()); assert tag==int(kind=='D')
        decoded_c=np.frombuffer(raws[0],dtype=np.uint8).reshape(600,192,256)
        if tag:
            b=np.unpackbits(np.frombuffer(raws[1],dtype=np.uint8),bitorder='little')[:28125].reshape(-1,3)
            decoded_table=(b * (1 << np.arange(3))).sum(1).astype(np.uint8).reshape(3125,3)
        else: decoded_table=None
        closed_path=root/(kind+'_'+order+'.decoded.u8')
        if not closed_path.exists():
            tmp=closed_path.with_suffix('.partial')
            closed=np.memmap(tmp,mode='w+',dtype=np.uint8,shape=F.shape)
            for t in range(600): closed[t]=predict(decoded_c[t],decoded_table,kind)
            h.apply_residual(raws[2],closed);closed.flush();del closed
            assert sha_file(tmp)==expected;os.replace(tmp,closed_path)
        sha=sha_file(closed_path);assert sha==expected
        B=packet.stat().st_size
        results.append({'kind':kind,'order':order,'packet_bytes':B,'field_sha256':sha,
            'exact_n600_decode':True,'within_gate':B<=94292,'archive_projection':60215+B,
            'score_claim':False,'axis':'macOS-CPU scorer-free exact packet bytes'})
facts=[h.et1.file_fact(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ('MANIFEST.json','RESULT.json')]
h.et1.atomic_bytes(root/'MANIFEST.json',json.dumps(facts,indent=2).encode())
h.et1.atomic_bytes(root/'RESULT.json',json.dumps(results,indent=2).encode())
print(json.dumps(results,indent=2))
# SO1_FIRST_RUNG_END
```

This remains a design sketch: the successor must pin its extracted source and wire the governed
done receipt/resource admission before launch. Those are **folded into the same first rung**, not
separate research projects. Counts checkpoints carry hashes and the input identity; decoded outputs
complete by atomic rename; coder stages retain all candidates and deterministic repeats. All stages
are deterministic and use no RNG. No code from this block was executed on F. Syntax/AST checking
of the block is not a scientific measurement or a runtime/resumability test.

Storage routing for that rung: source/metadata remain local, all materialized fields, table bytes,
residuals, coder alternatives, packets and decoded outputs go to its owned SSD root. Every payload
gets bytes/SHA in the manifest. Budget at least **A 2 GiB** above the currently enforced reserve;
the wrapper must measure actual resource needs and refuse if unavailable. No cleanup removes payloads.
Only atomic-write partial files may be resumed or replaced after complete reproducible state is
certified; unresolved partials are held. This design launches no process and creates no bulk artifact.

## Negative scope and what not to repeat

* **FORMULATION, existing HG1:** exact residual already overwhelms the generator saving. The
  18,900 renderer-bias edits do not replace its 1.33M fit errors. No unmodified HG1 re-fit is queued.
* **FORMULATION, LTG1/D3B/BLP1 instances as specified in their receipts:** another name for
  exact contours, sparse Lane indices, the same trunk grid, or the retained r10 Lane head adds
  no new mechanism. Their measured prices stay closed on those objects, not all future generators.
* **FORMULATION, REN2/RQ1 on current F:** refit/realization moves lose. RQ1's coarsening debts are
  far outside the measured repair exchange. SO1 holds their renderer fixed instead of reopening them.
* **FORMULATION-SCREEN, MC1/BD1/LS1:** temporal and finite-cell probability corrections do not
  authorize a demand-sized saving. LS1's Miller–Madow number is an estimate, not an information-
  theoretic lower bound. No repeated context-overlay or “oracle minus free table” rung is queued.
* **Inference closed:** a small native token error guarantees small distortion, a kept pose
  carrier guarantees kept pose after changing F, or the 2.18× form ratio is lossless. BZ2D/GF1
  directly contradict these premises. Exact reconstruction of F avoids having to transfer them.

## Handoff and boundaries

M this arm: current archive/header/field byte custody, source reads and hashes. D this arm: strict
budget, timing slack, sparse-edit support budget, scenario tables and explicit conditional grammar.
**Not measured:** coarse price, table price, B/D residuals, new argmax cells, renderer latency, new
archive/inflate, training, any CPU/CUDA score. No paid work, no Modal, no model/scorer launch, no
subagent delegation, no source/runtime/contract edit, no change to sealed trees or other arms' dirs.
No artifact was moved or deleted. The only scientific deliverable is this memo and its receipts.

Checkpoint identity is `ddm_so1`. Final serializer is attempted once, last, with post-edit hashes;
its receipt records actual rc. If it returns Git-object-denial rc 17, the verified bundle is the
handoff and MAIN owns landing. The pre-existing shared staged index is not edited by this arm.

## NEXT_IF_RESUMED

* **QUEUED-WITH-A-FIRE-ORDER** — owner **MAIN / SO1 byte-successor**; consumer store
  `/Volumes/VertigoDataTier/pact/ddm_so1_first_rung/RESULT.json`; fire trigger **harvest of this memo,
  revalidated field/pointer/source pins, an owned local byte-only claim and admitted storage/resources**.
  Bind the extracted source to the governed launcher and run the single B/D exact-packet
  rung above. If both exceed 94,292 B, fold this finite grammar; if one is at most 94,261 B, route its
  retained bytes to MAIN's receiver/timing build decision. No automatic scorer or training fire.
* **FOLDED into that rung's success branch** — owner **MAIN receiver owner**; consumer store
  `/Volumes/VertigoDataTier/pact/ddm_so1_first_rung/receiver/`; fire trigger **exact n600 packet pass
  at most 94,261 B**. Preserve the current renderer/carrier semantics in a new counted format,
  validate independent decode and the current timing policy, and only then follow the existing
  immutable candidate contract. This is not an additional measurement authorized to SO1 today.
* **QUEUED-WITH-A-FIRE-ORDER, conditional custody only** — owner **MAIN serializer landing owner**;
  consumer store `ddm_so1_20260912/LANDING.json`; fire trigger **serializer rc 17 with verified bundle**.
  Land exactly the declared hashes without sweeping unrelated shared-worktree files.

## LIVE-HYPOTHESES

* The current field's recurring fine patterns may be cheaply predictable from already decoded
  coarse spatial neighbors; a small counted table can represent jagged pre-distortion as well as
  smooth boundaries. Plausible because the information is boundary-local and the table covers all
  classes, but its actual residual geometry and coarse price are unmeasured.
* The fixed regularizer may beat the table after coding, even though its Hamming count cannot be
  better: clustered errors can be cheaper than fewer scattered errors. GDC2/GDC3 provide the
  reason to measure both in the same byte rung.
* A genuinely different Lane positional generator remains logically open only if its complete
  conditional representation pays topology and exact residuals. The concentration of Lane surprise
  makes it relevant; existing eight-dimension slogans and prior exact forms do not establish it.

## DEAD-ENDS

* Buying a 48 KB **exact** field by reusing HG1's lossy rate: GF1 already priced the missing
  residual at 385,448 B. Repeating this would repeat a measured formulation failure.
* Calling pass edits a standalone sparse entropy or adding them as unique sites: marginal coder
  deltas omit the new support grammar, subsets overlap, and the current footprint is a separate object.
* Removing a third of the stream solely because Lane owns a third of surprise: class attribution
  is not the quotient/occupancy factorization; D3B already measures the omitted cost.
* Repeating the known Lane contour/trunk/head constructions, current renderer refit/coarsening,
  or fixed-cell/temporal mixer screens without a changed mechanism: their receipts already price
  those formulations outside their bars.
* Importing 0.33 as a born n600 distortion, approximately 700 seconds as admitted receiver slack,
  or FB1's old perfect-axis bound as today's arithmetic: the later receipts and current components
  contradict all three. None is a valid successor premise.

composition S 0.13638261682704697 @ 179,111 B [contest-CUDA T4 n600] (move 48)
