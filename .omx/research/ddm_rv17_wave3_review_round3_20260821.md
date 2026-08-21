# ddm_rv17 — wave-3 review round 3 (the scoring round): 2 findings, counter RESET to 0/3

**Scope.** The wave-3 scoring round, items 1 and 4 (mine). Items 2 and 3 (instrument-cure
disposition vs F2–F12; the new comparator `experiments/ddm_fs3_same_instrument_pose_leg.py`)
are with arm `ad3653c5680247bda`, still running. **Nothing from that arm is adopted here.**
I re-verify arm findings at source before adoption — a standing rule this wave earned eleven times.

**Counter: 1/3 → 0/3.** Two findings below.

---

## ITEM 1 — the verdict memo (`e9df0d9b43`, `bdb2ad401c`, `9f502ba4a2`): PASS with one LOW finding

I scored it against my own rounds-1/1b convictions, with length-checked instruments
(`awk NR==n{print length($0)}`, never `cut` — my own round-8 law).

**The mirror headline is exemplary — this is my W3-F1 genus cured at the design level, not patched.**
L12–L20 carries, in the block a reader hits first:
- the measurement (`credit 664 B over 997 dropped tokens = 5.3280 bits/token`),
- the falsifier registered **before** the encode (5.2282 / 651.6 B),
- the survival **and its thinness** (`by 12.4 B, 1.9%`),
- the shape of that margin (`a TREND EDGE, not a plateau`) with the monotone decline printed
  in full (`7.19 → 6.12 → 5.95 → 5.56 → 5.328`),
- the instruction **`Quote the 1.9% with that caveat wherever it travels.`**, and
- **L20: `TASK #1176 TERMINAL VERDICT (§T11–T13): REFUSED on the MEASURED pose leg.`**

A caveat that names its own travel obligation is a stronger object than a caveat that sits
next to a number. That is the genus closed, not the instance patched.

**W3-F3 is dispositioned better than I argued it (L767).** I made a *structural* case: CONTROL B
and the realized totals are entailed by CONTROL A, so the triad is one check. The memo accepts
that and then supplies something I did not have — an **empirical existence proof**:
*"Round 2 passed this triad 38/38 and its row still died — that is the proof the triad is
reproducibility, not correctness."* An argument says the controls cannot establish correctness;
a measured row that passed every control and still died **demonstrates** it. I withdraw nothing
and gain the stronger form.

**W3-F8 is not merely addressed — it was pre-registered, measured, and settled (§T7, L856–861).**
The memo grants both halves of my finding (round 2 confounded an AVERAGE-over-a-REMOVAL with a
MARGINAL-over-an-ADDITION; the controls could not separate them), then builds the discriminator:
a removal of *marginal* tokens credits 5.3280, near the 5.9467 an addition of marginal tokens
costs, and nowhere near the 2.6573 a removal of a *whole set* credited. Verdict: **average-vs-marginal
is the real axis; round 2's attribution was correct; the alternative reading is measured false on
this lineage.** My reading lost on the evidence. That is the pre-registration working as designed.

### W3-F12 (LOW) — the round-2 block still attributes its conclusion to the wrong warrant

| field | value |
|---|---|
| **finding** | The ROUND 2 VERDICT block states an inference the memo itself has since disproven *as an inference*. |
| **file:line** | `.omx/research/ddm_fs3_jg5_real_price_reopen_20260820.md:7-8` |
| **evidence** | L7–L8: *"The re-screen's controls were perfect (38/38 pairs landed on the exact configuration the census predicted), **so** the census arithmetic was right and the PRICE TRANSFER was wrong."* The connective **so** makes the controls the warrant. L767 records that the triad is reproducibility only. L860–861 establishes the conclusion by the **mirror's measurement**, not by the controls. The conclusion is true; the stated reason is not the reason. |
| **why it is not dismissible as a historical record** | The block is **maintained, not frozen** — L6 already carries a withdrawal (*"Round 1's 7.11× is WITHDRAWN as a realized claim"*). A block that carries one correction can carry another. |
| **severity** | LOW. Number correct, verdict correct, warrant mis-attributed. A reader of the round-2 block alone learns a true fact by a false epistemology. |
| **verdict_scope** | INSTANCE |
| **proposed cure** | Replace the bare `so` with the actual warrant, e.g. *"…the controls were perfect (reproducibility, not correctness — see the W3-F3 disposition); §T8's mirror then measured the attribution correct."* One clause. |

---

## ITEM 4 — CF2 registrations (`e8388550d0`, `9395e5782c`): PASS on three legs, one MEDIUM finding

**`token_rate_model_direction_dependence_v1` — CLEAN, and it fixes the trap at the source.**
4.718 is registered as `JG1_LOGIT_RANKER_BITS_PER_TOKEN` and named in the module docstring as
*"``LogitPrice`` **RANKER** — ordering only, never a price, per its own docstring."* Ratios against
it are routed through a separately-named `ranker_relative_ratio`, so a consumer cannot reach a
ranker-relative number without reading the word *ranker*. The actual charged price (4.1379) is
annotated *"This is a price and may be a denominator."* The Series A / Series B split is preserved.
This is the wave-2 round-4 defect — my own — made structurally unrepeatable.

**`greedy_set_average_vs_marginal_price_v1` rate leg — CLEAN.** 5.3280 is derived in place
(`664*8/997`), the monotone decline is printed (`7.1862 → 6.1227 → 5.9451 → 5.5617 → 5.3280`),
and L629 refuses the misuse explicitly: *"carrying 5.3280 as a plateau value: it is a trend edge,
still falling when …"*.

**`src/comma_lab/packet_receipts.py` R15 — CLEAN, and cured the right way.** The known refusal is
documented verbatim in the module docstring (L7–L15): R15 serialized a `note` as a single-element
JSON **list** via a trailing-comma slip, append-only forbids editing R15, and the class note is
carried forward. It is not papered — L18 records that `__post_init__` now **refuses a non-string**,
so the writer cannot reproduce the slip. Documented defect **plus** a structural guard: the
two-landing pattern done properly.

### W3-F13 (MEDIUM) — the pose refusal does not travel into the registry

| field | value |
|---|---|
| **finding** | The greedy-set registration carries the mirror's **rate survival** and its trend-edge caveat, but **not** the row's terminal disposition. `grep -niE "pose\|#1176\|terminal"` over the module returns no hit that names the pose leg. |
| **file:line** | `src/tac/canonical_equations/ddm_cf2_token_price_laws_20260821.py:584-629` (registration body) |
| **evidence** | The memo's own headline binds the two: the mirror **survives on rate** (L14) and the row is **REFUSED on the MEASURED pose leg** (L20). The registry receives the first and not the second. |
| **why it matters** | The canonical-equations registry is a **live consumer surface** — it is queried by the planner, not read as prose. A consumer who pulls "5.3280, survives its falsifier by 1.9%" can read rate survival as **row admissibility**. It is not: the rate leg survived and the row died. |
| **the counter-argument, and why it does not fully hold** | A rate law is legitimately scoped to rate, and the pose leg is a property of the candidate row — so registering it could read as scope creep. But the memo has already ruled that the caveats on this number must travel *wherever it travels*, and the registry is the surface that travels furthest. CF2 cured the trend-edge half of that obligation and left the pose half behind. |
| **severity** | MEDIUM |
| **verdict_scope** | INSTANCE — the rate law itself is sound and I am not disputing 5.3280. |
| **proposed cure** | One clause in the registration's note: *"Rate leg survived its pre-registered falsifier; the row it came from (TASK #1176) was TERMINALLY REFUSED on the measured pose leg — the price stands, the candidate does not."* |

---

## Honest state

- **Counter: 0/3.** Two findings (W3-F12 LOW, W3-F13 MEDIUM). Both are one-clause cures on
  surfaces I do not own conflicts on; neither touches the packet or sealed custody. I filed
  them rather than fixing them because both edit documents under active MAIN authorship this turn.
- **Items 2 and 3 remain open** with arm `ad3653c5680247bda`. Its findings are not counted here
  and will be re-verified at source.
- **The substance has still never moved.** Across 20 + 11 + 3 rounds, no round has found a wrong
  score, a wrong pin, a wrong digest, a mis-scoped receipt, or an unverifiable archive claim.
  Every finding since wave-1 round 3 has lived in the apparatus that reports the work, not in
  the work. Wave 3 narrows that further: the arithmetic is sound, the mechanism adjudication is
  now **measured** rather than argued, and what remains is whether each true number carries its
  own caveats to the surface where a consumer meets it.
- **Both of this round's findings are the same shape** — a correct number whose accompanying
  obligation stopped one surface short. W3-F12: the warrant did not travel into the round-2
  headline. W3-F13: the pose refusal did not travel into the registry. That is my W3-F1 genus,
  and it is now the only genus wave 3 is still producing.

**Own-vehicle frontier: S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600]**
