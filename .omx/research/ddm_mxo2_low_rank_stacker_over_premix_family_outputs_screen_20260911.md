# ddm_mxo3 (resuming ddm_mxo2) — the pre-mix stacker screen, CLOSED at n600: the shipped collapse keeps nothing

**Axis:** `[macOS-CPU advisory / scorer-free n600 exact integer-row screen]`. No score claim, no Modal,
no scorer, no candidate archive. This screen's binding frontier, unchanged by it:
`composition S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600] (move 44)` — the archive every
number here was measured against. While this screen ran, a sister arm landed **move 45**
(S 0.1371383667388406 @ 180,246 B, archive `145e02e21f9a1cbc…`); that is not this arm's doing and does
not touch any measurement below.

## Answer

**The trigger does not fire. The timing half passes with 3.5–7.5× margin; the saving half fails by 323×.**
On the real n600 stream the best of the three ranks saves **9.280 B of 119,749 B** against a 3,000 B
threshold. The family is CLOSED at the scope stated below, and closed on a *ceiling*, not only on one
learner: a non-causal, held-out oracle over the same 23 pre-mix outputs also finds nothing — it **costs**
894–5,491 B, and all 23 families tested individually cost bytes too.

| rank | bytes saved | of 119,749 B | short of 3,000 B | host ns/symbol | T4 ns/symbol | T4 incremental | saving | timing |
|---|---:|---:|---:|---:|---:|---:|:--:|:--:|
| 2 | **+9.280** | 0.00775 % | 2,990.720 | 27.22 | 26.62 | +3.140 s | FAIL | PASS |
| 4 | −9.944 | −0.00830 % | 3,009.944 | 39.37 | 38.51 | +4.543 s | FAIL | PASS |
| 8 | −7.469 | −0.00624 % | 3,007.469 | 58.60 | 57.31 | +6.761 s | FAIL | PASS |

T4 projection factor 0.978073 (mxo1's measured leg ratio). Strict T4 receiver slack 27.581 s =
233.809 ns/symbol; the screen's own bar was 200 ns/symbol. Verdict recorded in `SCREEN.json`:
`CLOSE_FORMULATION`.

## The instrument reconciles (the falsifier that did not fire)

A COPY of the move-44 receiver was instrumented to emit, per coded symbol in the shipped 190-group
order, the 23 family probabilities, the mixer's own probability, the coded row and the decoded symbol.
The immutable move-44 tree was never touched.

| quantity | measured |
|---|---:|
| frames / symbols | 600 / 117,964,800 |
| ideal code length from the retained rows | 119,748.300 B |
| the REAL shipped token stream | **119,749 B** |
| framing difference | 0.700 B |
| relative | **0.000584 %** (ls1 standard: 0.0006 %) |
| decoded field vs the shipped field | byte-identical (`a92e7d90…`) |
| extraction wall clock | 2,705.3 s |

The per-frame rows were also checked against ls1's independently retained rows, element for element.
So the denominator in every number below is the real stream under the real coder, not a proxy.

## Why the zero is real, not a weak learner

A 0 B online result cannot by itself separate "there is no information" from "this learner is weak", so
the ceiling was measured separately (`experiments/ddm_mxo3_premix_oracle.py`).

**The question is binary, exactly.** MEASURED in `runtime/f26_corrector_native.c`: the whole 23-family
mix reaches the coded row through one scalar. `blended` forms `q = p_max·blended / (p_max·blended +
one_minus)`, the argmax entry becomes `q`, and every other class is rescaled by `(1 − q)/one_minus`. The
later within-miss reweight leaves the argmax entry alone. So the pre-mix machinery moves exactly
P(token = argmax), and its code length is the binary cross entropy of the shipped `q` against the
realised hit: **116,897.10 B, 97.62 % of the stream.**

The oracle partitions symbols on (hit class, shipped `q`, statistics of the 23 opinions) and scores each
cell at its own empirical hit frequency. Each family's opinion is its own log odds minus the mix's,
which cancels the shared base odds and leaves exactly the multiplier the mixer weights.

| partition | cells | occupied | in-sample net (shuffle-controlled) | **held out** | held-out recalibration | per-family held out |
|---|---:|---:|---:|---:|---:|---|
| summary, 1 block | 81,920 | 6,563 | +318.74 B | **−894.17 B** | −183.53 B | best −140.17, worst −258.40, **0 of 23 positive** |
| rich, 1 block | 1,966,080 | 51,444 | +1,115.62 B | **−5,203.96 B** | −119.34 B | best −963.03, **0 of 23 positive** |
| summary, 10 blocks | 819,200 | 45,296 | +882.42 B | **−5,491.37 B** | −183.53 B | best −140.17, **0 of 23 positive** |

Read this way:

- **Held out, the pre-mix table costs bytes.** Fitting on half the frames and scoring the other half —
  with KT back-off to the coarser table, so a finer partition is not punished merely for being finer —
  loses 894 B at best and 5,491 B at worst. Richer partitions lose *more*, which is what no signal looks
  like.
- **Even the shipped `q` cannot be recalibrated.** An empirical 64-bin refit of the shipped probability,
  fitted on 300 frames, loses 183.53 B against the shipped value. The mixer is already calibrated.
- **No single family hides in the summary.** Each of the 23 got its own one-axis held-out model. All 23
  cost bytes, in all three configurations. (These are 23 separate measurements reported as a range,
  never a total — disjoint gains do not add.)
- **The in-sample bound also clears the bar in the wrong direction.** Even letting the oracle score cells
  at frequencies fitted on the very symbols it scores, and subtracting only the shuffle control's
  overfit, the most generous partition reaches 1,115.62 B — still 2.7× short of 3,000 B.

**The Q15 boundary is not the limitation.** 80,922,120 symbols (68.6 %) have all 23 families rounding to
the same Q15 value as the mix, so the surface cannot show a disagreement there. Those symbols carry
**94.44 B, 0.0808 % of the binary code length** — because the bits live where the families *do* disagree.
Even granting a perfect oracle on every blind symbol, the total ceiling is 1,115.62 + 94.44 =
**1,210.06 B**, still 2.5× short of the trigger.

## The mechanism (and a charter premise that does not hold)

The mxo2 charter described the collapse as "a fixed (counted 35-weight) mixer". **Neither adjective is
true**, and the correction is the explanation for the null. MEASURED in the shipped source:

- Each family emits a Krichevsky–Trofimov multiplier clamped to `[1/16, 16]` (`ODDS_LOW`/`ODDS_HIGH`).
- They combine multiplicatively in odds space, `blended = Π m_pos^(w_pos / 2^6)`, over
  `N_MIXER_CONTEXTS = 4,000` contexts (class × boundary × agreement × homogeneity × 8), each holding its
  own 23 weights — 92,000 weights, not 35.
- Those weights are **cold-started**: member 0 (`shipped_joint`) at 1.0, all others at 0, in every
  context. They cost **zero counted bytes**.
- They are **learned online**, every decode group, from the residual against the quantised stretch
  (`LR_SHIFT 24`, count-normalised).

So move 44's receiver already *is* an online, context-conditioned stacker over the 23 pre-mix outputs.
mxo3 stacked a second, nonlinear one on top of it and measured zero, three ways. That is the honest
reading: the collapse is not a lossy projection that discards information — it is an adaptive mixer that
has already taken what the families carry.

## Scope of the negative (verdict ladder)

CLOSED, at **family scope**, for a decode-time learner reading the 23 pre-mix family outputs:

1. the online integer rank-2/4/8 random-projection quadratic stacker, measured on realized code length
   over the full n600 stream;
2. any causal learner over the summary statistics of the 23 opinions (spread, signed mean, peak,
   crowd direction) — bounded held out at two baseline resolutions and with per-block refitting;
3. any causal learner over a single family's own opinion — 23 held-out models, none positive.

NOT closed by this work: a learner reading a context the 23 families do not span (tc2/tc3/ls1 territory,
already measured separately). The one seam inside this family is **specific pairwise structure**: the
screen's rank-8 kernel covers it with only 8 random quadratic projections of a 253-dimensional pair
space, and no oracle bound was built for named pairs. Given that every marginal and every summary is
negative held out, a pair-only effect worth 3,000 B would have to hide from all of them; that is the
honest remaining gap, and it is small.

## What would reopen it

A measured, **held-out** positive gain from a feature of the 23 pre-mix outputs that is not a function of
(spread, signed mean, peak, crowd-direction, single-family opinion) — in practice, a named family pair.
The instrument takes it as one more axis and one more pass over the retained surface, at $0.

## Provenance

- charters: mxo3 `b34fbba20c6b1bf7…`, mxo2 `ad5fb483795450d4…`
- mxo1 memo `60b5a34bb6146af8…` (27.581 s slack, 233.809 ns/symbol, the 3,000 B / 200 ns trigger)
- ls1 memo `b729faa146b62ea3…` (the 0.0006 % reconciliation standard)
- tc1 memo `d6f09422a1c0896e…`
- pointer move 44: commit `99625f32f`, archive
  `04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e`, 180,406 B
- shipped receiver source `runtime/f26_corrector_native.c` `3e2705f550503612…`; the instrumented copy
  `2dd4fe32861819a0…`
- code: `experiments/ddm_mxo2_low_rank_stacker.py`, `experiments/ddm_mxo2_stacker.c`,
  `experiments/ddm_mxo3_premix_oracle.py` and their tests (commits `ac4208093`, `01ef120c6`)

## Retained payloads (SSD tier, nothing discarded)

Root `/Volumes/VertigoDataTier/pact/ddm_mxo2_low_rank_stacker_v3/`:

| artifact | size | sha256 (prefix) |
|---|---:|---|
| `SURFACE.json` | 140,617 B | `a857f0f9456804b0` |
| `SCREEN.json` | 6,398 B | `cfeb0c9289624c44` |
| `BENCHMARK.json` | 723 B | `7bba8ac9cb4af87f` |
| `ORACLE_summary_b01.json` | 2,143 B | `a8bb9b361993290f` |
| `ORACLE_rich_b01.json` | 2,153 B | `61114ad2b465597e` |
| `ORACLE_summary_b10.json` | 2,146 B | `52304cdf5546f9e4` |
| `BINDING.json` | 31,231 B | `b198ef79675ce55c` |
| `surface/frames/` (600 npz, each with its own hash receipt) | 7.7 GB | per-frame `.json` |
| `surface/decoded_tokens.u8` | 117,964,800 B | `a92e7d902a449896…` |
| `screen/checkpoints/` (600 states; final `state_0600.npz`) | 4.7 MB | `d52bdbc4bee2b492` |
| `native_stacker/libmxo2_stacker.dylib` | 40 KB | `6bc96cfd1dd7801a` |

The earlier `ddm_mxo2_low_rank_stacker/` and `_v2/` roots are left in place with their partial surfaces;
nothing was deleted.

## Two apparatus notes

1. **The resume blocker.** mxo2 could not be restarted at all: `prepare()` compared freshly built pins
   (tuples) against `BINDING.json` (lists), so every run after the first raised "source/input binding
   drift" on unchanged inputs. Pins are now JSON-normalised before the comparison, and the owned root is
   adopted from `--resume-from` under a family prefix guard, so a new immutable release no longer needs a
   new constant. Both are covered by tests.
2. **The custody layer earned its keep.** A one-line hardening of `adopt_root`, added *after* the surface
   was extracted, made the running source differ from the pinned `source_release`, and the screen refused
   to start. The change was a no-op on this path (the tier is not a symlink), so it was withdrawn rather
   than worked around — the source that produced the artifacts and the source that reads them are the
   same bytes again.

<!-- # FORMALIZATION_PENDING: a measured null needs no new law; the realized-bytes law lands in the equations leg with the next exact row -->
