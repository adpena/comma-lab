# ddm_op1r — EXAMINE ALL SIGNAL → PROPOSE THE OPTIMAL PATH FORWARD

Operator directive 2026-08-09: *"examine all signal and propose optimal path forward taking as
long as it needs with full authority to do everything."* You have FULL AUTHORITY: research
online, read the whole corpus, run measurements, build, commit, race, refute. Take as long as
you need. There is no page budget and no time budget on your reasoning.

You are NOT a crosswalk arm and NOT a convocation. You are the agent who decides what we
actually do next, and proves it from measurements.

## THE FRAME (operator-binding, 2026-08-08, supersedes prior framing)

**PR130 IS THE BASE.** Operator verbatim: *"take fresh eyes to pr130 and then optimizing further
with the stuff we have and they don't — that is the roadmap."* Also: *"KT is dead."* Also:
*"don't be too obsessed with the rate ... all of that should match pr130 or better."*

Our own line (tq1c, S 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]) is the thing to get
UNSTUCK FROM, not the thing to improve. We start AT 0.1721417 and go DOWN.

Canonical memo: `.omx/research/ddm_rm1_20260808/PR130_BASE_ROADMAP.md` (commit dde2a650b3).
Hot state: `.omx/state/main_hot_state.md` NEXT_BOUNDARIES (read the whole file, it is truncated
in session hooks and MAIN has been burned by exactly that).

## MEASURED STATE YOU INHERIT (re-derive everything; do not trust this list)

PR130's triple, from THEIR OWN 600-sample CUDA report
(`repro_repo/evidence/source_archive_official_ada_report.txt`):
d_seg 0.00028609 (0.028609) · d_pose 0.00001967 (0.014025) · rate 191,052 B (0.127214) → 0.1721417.
Both distortion axes already crushed; ~74% of their residual is RATE.

PR130 archive anatomy, MEASURED by parsing `releases/cpr1/archive.zip` (one ZIP_STORED member `p`
= u32(len(models)) ‖ LZMA-XZ(semantic ‖ carrier ‖ hpac) ‖ tokens):

| object | bytes | share |
|---|---:|---:|
| tokens (raw HPAC-coded) | 116,980 | 61.3% |
| semantic_blob (renderer weights) | 40,252 | 21.1% |
| carrier_blob (pose basis+coeffs) | 23,054 | 12.1% |
| hpac_blob (AR prior model) | 20,179 | 10.6% |
| ZIP overhead | 100 | 0.05% |

Label-stream pricing under PR130's OWN HPAC coder — **the fact that just reframed the vehicle
question, and the reason this arm exists**:

| payload | bytes | state |
|---|---:|---|
| VEH labels (tq1c, our renderer's argmax) | 112,044 | COMPLETE, exact decode verified (hb2) |
| GT labels (actual `lstars`) | 135,732 | ep54/60, killed mid-run, checkpoint survives |

GT costs **+23,688 B = +0.015773 S**. But that premium must be weighed against what GT BUYS, and
the operator supplied the missing half from memory (2026-08-09): *"We did a long VEH HPAC run and
the rate was great but our d_seg plateaued at 0.004 ... and you said our token stream was not
optimal compared to PR130."* Confirmed on disk — that is the tq1c plateau, **d_seg 0.004305420**.

**The VEH label stream is cheap because it is our renderer's own argmax — self-consistent,
low-entropy, and WRONG.** Its seg term is 0.430542 S against PR130's 0.028609 S. The correct
comparison is therefore:

| quantity | value |
|---|---:|
| GT rate premium | +0.015773 S |
| VEH→PR130 label-error gap | **0.401933 S** |
| ratio | **25.5×** |
| break-even: GT realization d_seg must be ≤ | 0.004147691 |
| i.e. GT need only beat the VEH plateau by | **3.66%** |

**MAIN's earlier framing ("GT must halve their seg term to break even") was WRONG** — it compared
the premium against PR130's seg term instead of against the seg improvement GT buys over VEH.
Corrected here. The operator's standing commitment (GT + learned prior) is **supported** by this
arithmetic, not threatened by it.

What remains genuinely open and is YOURS to adjudicate: the 0.401933 is a CEILING (perfect
realization). Shipping GT labels gives d_seg = REALIZATION error, which is unmeasured. The
break-even is only 3.66% better than the VEH plateau, so the bar is low — but *measure or bound
it*, don't assume it. And the deeper reading the operator's recollection implies: our token stream
is the wrong FAMILY (latent IX2TOK01 vs PR130's dense semantic), so the plateau may be a family
property, not a training-budget property — #978 names this and it was never given a full-vehicle
race.

Live resumable asset: `/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/checkpoints/gt/hpac_selfcompress_e60.latest.pt`
(ep54, weights-only; past the ep30 line so warm-start is legal but must be labelled
FORM_DEVIATED_RESUME per hb1's own caveat). Driver log with the full epoch trace:
`/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/driver.log`.

**THE UNLOCK, measured today:** PR130's `inflate.py:663-669` hard-requires CUDA
(`raise RuntimeError("...requires the official GPU rail")`, `device = torch.device("cuda")`), but
every internal function is device-parameterized (`coordinates(batch, device, dtype)`,
`load_hpac(raw, device)`, `decode_tokens(hpac, bytes, device)`, `render_video(..., device)`); only
`torch.cuda.empty_cache()` needs guarding. Our dependency closure is COMPLETE: constriction ✓,
torch 2.12.1 ✓, numpy ✓. **Their entire vehicle is one gate from running on our machine.**
Caveat: their 0.1721417 is CUDA; a local run is `[macOS-CPU advisory]` and will differ (the
long-open CPU/CUDA GT drift, #906). It gives a working base and RELATIVE deltas, not their number.
Their repro tree is a public-PR intake clone — **no in-place edits**; patch in our tree.

**Off-the-shelf grant is LIVE** (operator 2026-08-06,
`pr130_code_off_the_shelf_authorized_20260806`): PR130 repro trainers / HPAC / carrier / repack are
directly reusable on our line. NO-FAKE #7's prohibition half is lifted for this lineage; the
honesty half (borrowed_substrate_accounting + attribution) is UNCHANGED.

## THE QUESTION THAT WAS OPEN — now RESOLVED by operator recall, verify it at source

Operator (2026-08-08): *"We already ran that, I thought. VEH. And the results weren't good."*
Then (2026-08-09): *"We did a long VEH HPAC run and the rate was great but our d_seg plateaued at
0.004 ... and you said our token stream was not optimal compared to PR130."*

So "not good" was never the RATE — it was the **d_seg plateau at 0.004305420** and the diagnosis
that our token FAMILY (latent) is wrong versus PR130's (dense semantic). MAIN had been quoting the
VEH byte number as if it were a favourable result; it is favourable on one axis and disqualifying
on the other. **Verify this at source** (tq1c receipts, the burn plateau telemetry, #978) rather
than inheriting it — but the routing consequence is already large: a cheap label stream that
plateaus 15× above the bar is not a vehicle, it is a rate datum.

## WHAT WE HAVE THAT PR130 DOES NOT (the "optimize further" half — PRICE THEM, don't assume)

All PROJECTED / scoped to a RETIRED vehicle unless you re-measure. Treat every one as a
hypothesis needing a same-object price on a PR130-class payload:

- **Rate:** #869 adaptive per-cell token waterfill (−113,555 B projected, IX2TOK01-scoped) ·
  CR1 edge-conditioned support coder (−110,538 B / −19.221% on its selected edge object) ·
  SMEVR (WINS phase-field streams 15–20%; LOSES the IX2TOK01 token bulk +5,183 B — races, not
  reputation, per #940) · cell-drop waterfill knee (−0.0983 seg+rate, banked + exact-evaluated) ·
  #933 ±1.0 token range literal.
- **Weights:** #311 TropNNC · #336 sensitivity bit-allocation · #157 KKT reverse-waterfill ·
  #140 low-rank pose codec (2.7×). None applied to PR130's 40,252 B renderer or 23,054 B carrier.
- **Pose:** od9 cheap pose carriage (40,444 B projected) · the joint/in-loop descent family
  (#366, FAMILY-OPEN) · #850 the pose GN solve is TRUNCATED (hard-capped 2–3 relins, no
  convergence test, still descending 13–23%/iter when it stops).
- **Wall-clock:** our whole MLX/Metal stack — custom kernels, fused-R, grouped-backward,
  mx.compile. Unscored, but it multiplies every lever by races-per-day, and their chain needs
  CUDA+DALI. NONE of it is in the borrowed torch trainer.
- **Seg:** the entire seg corpus. Our own d_seg is 15× worse than theirs, so on seg we ADOPT
  first and improve second.

## WHAT YOU OWE

`.omx/research/ddm_op1r_20260809/OP1R_PATH.md`:

1. **THE PATH.** One ordered, concrete plan. Each step: what runs, what it costs, what it
   produces, the falsifier that kills it. Not a menu — a decision, with the losing options named
   and why they lost.
2. **The vehicle adjudication.** GT-primary vs VEH-primary vs hybrid, decided on the measured
   rate premium AND whatever you can measure or bound of the d_seg half. If you cannot decide,
   name the single cheapest measurement that would, and run it if you can.
3. **The MATCH step, executed if you judge it right.** Getting PR130's decoder running locally is
   step 1 of the operator's roadmap and nothing has executed it. You have authority.
4. **Per-lever prices** on PR130-class objects for anything you promote — measured or explicitly
   PROJECTED with the projection's basis named.
5. **Honest labels on every number**: MEASURED / DERIVED / INFERRED / ASSUMED / PROJECTED.
6. **`NEXT_IF_RESUMED:`** line.

## OPTIMAL FORM

- **Reference form:** a principal engineer with full repo history, the competitor's complete
  source, and authority to run anything — deciding what to build next and proving it.
- **Scope reductions (legal):** you may bound n, epochs, or sweep breadth; say so per row.
- **Mechanism reductions:** NONE without an explicit TOY-BRACKET declaration that says the row
  cannot produce a family verdict. A projection presented as a price is the fake this charter
  most needs you to avoid.
- **Provenance pins:** PR130 repro repo HEAD `e34f31bc4969042c0051ac81aa3c56884419a231`
  (`src/tac/pr130_lift/__init__.py::SOURCE_REPO_HEAD`; `LIFTED_AT_HEAD` is a DIFFERENT quantity —
  our copies' origin — do not collapse them). Roadmap memo commit `dde2a650b3`.

## RULES

- **RECALL FROM DISK, NEVER FROM WORKING MEMORY.** `.venv/bin/python tools/recall_fused.py
  "<query>" --top N` (positional query). MAIN violated this repeatedly today and it cost a
  16-hour run. Read `main_hot_state.md` IN FULL.
- **Re-derive, don't confirm.** Quote file:line. `UNDETERMINED` with a named missing input beats
  a guess. A zero is a measurement — check the instrument before believing it (quote your globs;
  an unquoted `--include=*.py` under zsh returns a false clean 0).
- **Never invent** CLI flags, API names, or VALUES. Grep `add_argument` / `def` first.
- **Kill process GROUPS, not wrappers** (`kill -TERM -<pgid>`). MAIN killed a wrapper today, the
  child reparented to init and burned 297% CPU for 24 minutes while being reported as dead.
- **ONE Metal fire at a time**, through the governed path. Check for live trainers before
  launching anything — there were four running today and two were orphans nobody knew about.
- MPS/MLX are NEVER score authority — label `[macOS-CPU advisory]` / `[macOS-MLX research-signal]`,
  `score_claim=false`. Only `upstream/evaluate.py` on exact archive bytes is a score.
- Upstream snapshot is READ-ONLY. Public-PR intake clones are READ-ONLY.
- No `/tmp` in any persisted evidence. Bulk artifacts → `/Volumes/VertigoDataTier/pact`.
- Commit via `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256` per
  file, tags `[no-triality] [p0-ledger-ok]`. `.py` needs two `review_tracker.py mark-file` passes;
  `REVIEW_GATE_OVERRIDE=1` is FORBIDDEN with `.py`.
- **NO Claude/AI attribution and no Co-Authored-By trailer on any commit.** Commits are the
  operator's alone.
- n≥120 stratified-random, never a prefix (m88/m96: prefix bias INVERTS by axis — pose prefixes
  measure 2.5–4.2× HARDER, seg ≈0.96× easier, rate ≈neutral).
