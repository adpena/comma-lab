# CONTRARIAN + HOTZ position memo — the small-basis Track-A DAG is rearranging deck chairs

**Seat:** Contrarian + George-Hotz. **Charter:** challenge the FRAMING, surface the highest-EV ignored
move, never conserve. **Date:** 2026-06-16. **Mission (non-negotiable):** lower the EXACT score below
0.15. **Frontier UNMOVED: 0.19109982 [contest-CPU], 177,169 B** (`canonical_frontier_pointer.json`).

All numbers below are grounded in the repo (`frontier_pointer_move_ledger_20260610.md`,
`sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`, `small_basis_sub015_build_plan_20260615.md`,
`track_a_long_train_then_taper_capacity_roadmap_20260616.md`, the running PID 74413, and the S-formula
arithmetic I recomputed).

---

## 0. The one-paragraph verdict (read this first)

**PIVOT.** The proposed DAG (long-train base_ch=20 → taper → capacity-last → byte-close → exact) is the
SLOWEST credible path to an exact row, and its sub-0.15 reachability rests on an FP4 step the build
plan's OWN smoke already falsified (`+0.2531 ΔS`, post-hoc FP4 NO-GO). At the int8 rate the small basis
actually achieves (~90 KB), even the *dream* d_seg=0.0006 lands **S≈0.1516 — ABOVE T_3** (my arithmetic
below). The vehicle is fine; the *ordering and the rate story* are wrong. The single highest-EV move the
program is NOT doing: **land the C8 bilinear-skip archive-export + oracle-parity gate** so that ANY of
these training runs can become a byte-closed exact row at all — that is the actual binding constraint,
and it has been a NotImplementedError blocker since 2026-06-10 (#81 audit) while we polished training.

---

## 1. The framing error: we are optimizing a vehicle 2× worse than the frontier, and the gap is mostly RATE we can't pay

The base_ch=20 basis scores **~0.39 advisory** (96-pair). The frontier is **0.191**. The session's
celebrated result — d_seg dropping 0.00359 → 0.002786 on a power-law — is real, but let's do the
arithmetic the DAG never closes (S = 100·d_seg + √(10·d_pose) + 25·B/37,545,489):

| scenario | bytes | d_seg | d_pose | **S** | seg / pose / rate |
|---|---:|---:|---:|---:|---|
| frontier (current) | 177,169 | 6.7e-4 | 3.4e-5 | **0.203*** | 0.067 / 0.018 / 0.118 |
| small basis, int8, advisory d_seg | 90,000 | 2.79e-3 | 1e-4 | **0.370** | 0.279 / 0.032 / 0.060 |
| small basis, int8, **dream** d_seg=0.0006 | 90,000 | 6e-4 | 1e-4 | **0.152** | 0.060 / 0.032 / 0.060 |
| small basis, **FP4** dream (the plan's claim) | 60,000 | 6e-4 | 1e-4 | **0.132** | 0.060 / 0.032 / 0.040 |

(*the 0.203 row uses my recomputed component split; the pointer's banked 0.191 is the recoded-R3 entropy
bolt-on — see §3.)

**Two things jump out:**

1. The small basis only crosses sub-0.15 in the **FP4 row**. But `small_basis_sub015_build_plan_20260615.md`
   line 60, the plan's OWN measured smoke, says: **"VERDICT: post-hoc FP4 NO-GO"** — d_seg +56%, d_pose
   +588%, net `ΔS +0.2531`. FP4 "requires FP4-QAT (train weights to be 4-bit-robust), NOT a free codec
   swap." So the rate maker that the whole sub-0.15 claim depends on **does not exist** and is itself a
   from-scratch retrain gated on research. The reachability line ("d_seg=0.0006 + FP4 + FiLM ⇒ S≈0.105")
   in the build plan header is **already partially falsified by the file's own line 60.**

2. At int8 rate (the rate we can actually pay today), reaching sub-0.15 needs d_seg=0.0006 AND we're at
   0.002786 — a **4.6× gap** the power-law is closing slowly (0.00359→0.00279 took +150ep). And the
   "dream" int8 row is **0.152 — still above 0.15.** The margin is gone before we even start.

**This is the deck-chairs critique, quantified.** Long-training the small basis chases a d_seg target
that, even if hit, lands at-or-above T_3 unless an unbuilt FP4-QAT retrain materializes. The DAG defers
capacity (the one lever that buys real d_seg headroom) to LAST, and defers the rate story to a step its
own evidence says is NO-GO.

---

## 2. The ordering is backwards, and "long-train the small basis" is deferring the real question

The DAG order is: long-train → taper → **capacity LAST**. But the binding constraint at the frontier is
**rate**, and the only lever that lowers rate without dying on distortion is a SMALLER basis that still
holds d_seg — i.e. **capacity is the question, not the afterthought.** Taper is explicitly
"rate-neutral" (build plan line 63), so it cannot move the rate term at all. Long-training base_ch=20
answers "how low can THIS fixed-capacity basis push d_seg" — a number that the table above shows is
**not low enough at the rate it costs.** That is the definition of optimizing the wrong variable.

The repo already knows this. `frontier_pointer_move_ledger` row #74 (the live exception) says the open
class is "score-domain RETRAINED smaller renderer … the funded-long-train **asymptote on the pose term**"
and the 40 KB student already reached d_seg 3.44e-3 / d_pose 2.43e-3 on a DESCENDING curve. The
highest-leverage unknowns are (a) the d_pose asymptote of a small student and (b) **whether a working
archive can even be built from a skip-on run** — NOT one more decimal place of base_ch=20 d_seg.

---

## 3. The frontier itself is a bolt-on — and the cheapest exact-row mover may be ANOTHER bolt-on, not a retrain

The current 0.191 pointer is **`lane_pr110_payload_entropy_recode`** — an entropy-recode of borrowed
PR#112 substrate (`defensive_bank=true`, `borrowed_substrate=true`, submission-blocked on the
`constriction` allowlist). It is NOT our HNeRV. **The leaderboard winners (PR95/100/101/103) won by
bolting ≤350-LOC coders onto a verified substrate** (CLAUDE.md HNeRV-parity L7; the 241-LOC silver
lesson). We have a verified-bytes 0.191 substrate sitting right there.

The DAG plan's OWN THREAD A names the cheapest open class: a **score-aware ADDITIVE adapter on the frozen
frontier**, break-even **a 1 KB adapter needs only −1.0% d_seg** (or −7.4% d_pose). That is a far smaller,
faster credible move than a multi-week from-scratch base_ch=20 campaign. The 8-direction convergent
finding (#64/#69/#71/#72/#73/#54/#79/#52) closed POST-HOC *re-coding* of the frozen bytes — but it
explicitly leaves **ADDITIVE adapters (Thread A) and RETRAINED smaller bases (#74) OPEN.** The DAG picks
the slowest of the two open classes.

---

## 4. Means-vs-ends: this session is the named failure mode

CLAUDE.md's ANTI-SIGNAL-LOSS rule and the sub-0.15 firewall both name "means-hoarding" as a
mission-level NO-FAKE violation. Look at what this session produced (`ls .omx/research`): **15 rounds of
`layer2_levers_review_round*`**, an oomph disambiguator, FiLM-warm-start SEALs, harness SEALs, taper
carriers, completeness ledgers, underpower audits. Enormous, disciplined, honest engineering — and the
exact pointer is **UNMOVED at 0.191.** The running process (PID 74413) is
`launch_oomph_finetune_disambiguator.py --arm oomph` — an *advisory 96-pair disambiguator*, the means,
not an exact-row candidate.

**When does this DAG produce its FIRST exact-eval row that could cross a threshold?** By its own phasing:
after #1 (running, ~5.6h) → #2 long train (the plan says ~16h at 96-pair, but a real candidate needs
600-pair from-0, "~6× slower/ep" + "local n600 ≈ 5-6 months", per the DAG C4 node) → #3 taper A/B → #4
capacity scaling → THEN byte-close → THEN exact. **The first exact row is weeks-to-months out, and it's
gated on a byte-close step (C8) that is currently a NotImplementedError.** That is the leapfrog setup the
operator explicitly extincted.

---

## 5. THE biggest single flaw: the C8 export blocker means NONE of this can become an exact row

`frontier_pointer_move_ledger` #81 audit, 2026-06-10, NEW BLOCKER C8:

> "use_bilinear_skip=True raises NotImplementedError in archive export (mlx_renderer.py:7456) — a
> successful skip-on run CANNOT build a contest archive until export+oracle-parity lands."

The root-cause analysis (#75/#76) is unambiguous: **skip-free PixelShuffle+sin = mean-field blur →
argmax collapse.** The working d_seg-descending loop (0.508→0.081 in 8 CE epochs, #76) needs the
**bilinear-skip** decoder. But bilinear-skip **cannot be exported to a contest archive.** So either:

- the small basis is skip-FREE (then it inherits the mean-field d_seg ceiling — explaining why 0.002786
  is so hard to push), OR
- it's skip-ON (then it **cannot be byte-closed** and can never produce an exact row).

Either way, **the byte-close path is blocked or capped, and the DAG step that matters (byte-close → exact)
is the one with zero engineering invested in it this session.** This is the binding constraint hiding
under a pile of training polish. (Note: the #81 line cites a large substrate renderer at line 7456; the
clean `src/tac/mlx_renderer.py` is 904 lines and uses bilinear-upsample+Conv2d, skip-transposed-conv.
The export-parity status of the ACTUAL skip-on basis used by the campaign must be verified before any
long run — if it's the same blocker, the long run is pre-doomed to advisory-only.)

---

## 6. VERDICT + the ONE move

**VERDICT: PIVOT (with a REORDER fallback).**

**Single biggest flaw in the DAG:** it invests weeks of training into a vehicle whose sub-0.15
reachability depends on an FP4 step its own smoke falsified, while the actual binding constraint — the
**C8 bilinear-skip archive-export + oracle-parity gate** that turns ANY skip-on run into a byte-closed
exact row — has zero work allocated and remains a NotImplementedError. We are polishing a car with no
road to the finish line.

**The ONE move I'd make FIRST (cheapest exact-row mover, ships in days not months):**

> **Build the C8 export+parity gate AND, in parallel, dispatch a Thread-A score-aware ADDITIVE adapter
> against the frozen 0.191 frontier — targeting the −1.0% d_seg / −7.4% d_pose break-even — then
> byte-close it and run the dual CPU/CUDA exact eval.**

Why this and not the long train:
- It produces a **real exact-eval row in days**, against a substrate we already have verified bytes for
  (0.191), instead of weeks-to-months out against an unbuilt vehicle.
- A 1 KB adapter at break-even is a **241-LOC-silver-class move** (smallest credible bolt-on), exactly
  the velocity pattern CLAUDE.md "Race-mode rigor inversion" mandates.
- C8 export-parity is the **shared dependency** every retrain path (#74, capstone #78, the small basis)
  also needs — building it once unblocks the slow paths too, so it is never wasted even if the adapter
  misses.
- It is measurement-first: it forces the FIRST byte-closed exact row this whole campaign has been
  deferring, which per the DAG's own §1 is "the currency."

**If the council insists on the small basis, then REORDER:** capacity-FIRST (jump base_ch20→36 at
600-pair from-0 as the baseline, since rate is the binding term and capacity is the only rate-headroom
lever), build C8 export-parity CONCURRENTLY (not last), and DROP the FP4 step from the reachability claim
until FP4-QAT is actually demonstrated to hold d_seg (the build plan's own gate). Do not run a 16h
96-pair long-train whose advisory d_seg "win" cannot become an exact row and whose sub-0.15 arithmetic
(int8 dream = 0.152) is above threshold before it starts.

**What I am NOT saying:** I am not saying kill the small basis (Forbidden premature KILL — it's the #74
OPEN class, and its d_seg descent is real signal). I am saying: stop letting "long-train the small basis"
stand in for the two questions that actually gate sub-0.15 — **can we byte-close a skip-on run (C8), and
is a cheap additive adapter on the frontier a faster exact row than a from-scratch retrain.** Answer
those with MEASURED rows first; the long train is the means, an exact pointer move is the end.
