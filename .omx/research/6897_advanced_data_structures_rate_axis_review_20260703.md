# MIT 6.897 Advanced Data Structures (Demaine, Spring 2003) → OUR RATE-AXIS coding toolkit — deep review for the Lever-D flicker-residual coder (#279)

**UTC** 2026-07-03 · **authority** `[$0 read-only research / advisory]` · **pointer UNMOVED contest-CPU 0.19110** ·
**score_claim** false · **promotable** false · **ready_for_exact_eval_dispatch** false. This is a MEANS (a
rate-coding toolkit + a concrete Lever-D coder recommendation). It moves no exact row by itself. Every rate
number below is BYTE-REASONED with its mechanism and labeled `ESTIMATE` unless it cites a byte-closed artifact;
no rate claim is asserted — the #279 build must MEASURE it byte-closed on the real residual.

**Operator ask (2026-07-03):** read the actual scribe-note PDFs (not the index), map 6.897 onto our contest,
review the OSS + licenses + inflate.py portability, and read the follow-ups/critiques (the modern state beyond
2003). Honor RESEARCH-DEPTH: read beyond the abstract, review the OSS, close the false-friend doors.

**Scoping honesty up front (NO-FAKE):** rate is **NOT our binding wall** — d_seg is (the lane/flicker floor,
per `canonical_research_index_rate_20260629.md` §2 and the d_seg index). These are RATE-axis tools. Their
highest value RIGHT NOW is **minimizing the COUNTED bytes `b` (per-flip byte cost) of the Lever-D residual
sidecar**, which lowers its break-even survival `σ* = b / 1.273108` and improves its net-S — and, at the margin,
could flip the measured NO-GO → GO **iff** the joint temporal+spatial conditional entropy of the residual sits
below the GO threshold (a byte-measurement, not a guarantee these tools can manufacture; see §6).

---

## 0. What I actually read (source PDFs, extracted with `pdftotext -layout`)

Index: `https://courses.csail.mit.edu/6.897/spring03/scribe_notes/` (19 lectures). Read in FULL: **L13
Compression**, **L12 Succinct data structures**, **L10 + L11 Suffix trees/arrays + RMQ/LCA**. Read the key
claims of **L1/L2/L4** (van Emde Boas / fusion trees) and **L14/L16** (cache-oblivious) to close the
false-friend doors. The 2003 PDFs are dvips/Type-1; WebFetch cannot decode them — `pdftotext` was required
(a real read-beyond-the-abstract, not the index).

---

## 1. THE COMPRESSION CORE — L13 (the bzip2 lineage; THE tools for the residual)

The exact results (verbatim math from the scribe note, Demaine/Liang/Malan, Apr 9 2003):

- **Huffman (Thm 1–2):** optimal prefix code; `E[|w_i|] ∈ [H, H+1)`. The **integer-bit penalty is
  MULTIPLICATIVELY catastrophic at low entropy** — their worked example `Σ={a,b}, p_a=1/1024`: `H≈0.01` but
  Huffman spends 1 bit/letter (~100× entropy). **Consequence for us:** a per-flip Huffman/fixed code on a
  low-entropy residual symbol (a flip that is *almost always* the same GT class given context) wastes ~1 bit
  where the true cost is ≪1 bit. Huffman is the WRONG coder for the highly-skewed conditional distributions
  the flicker-residual has.

- **Arithmetic / range coding (Thm 3):** encode the whole string as one real in `[0,1]`, recursively
  partitioned by the probabilities. `E[total code length] ∈ [Hn, Hn+1)` — **overhead is ~1 bit over the ENTIRE
  string, not per symbol.** This is exactly the fractional-bit aggregation the low-entropy residual needs. **We
  already ship this** (`src/tac/lossless/range_coder.py` `RangeEncoder`/`RangeDecoder` + `normalize_probabilities`
  + `cumulative_frequencies`; and `constriction` is already an in-tree dep — `pr91_hpm1_range_contract.py`,
  `hnerv_pr103_lc_ac_schema.py`, `pr106_latent_score_table.py`). PR103-silver already used
  `constriction.stream.queue.RangeDecoder` per L30.

- **Higher-order (empirical k-th-order) entropy `H_k`:** `H_k(x) = Σ_{|w|=k} Pr[w] · H_0(successors(w))`.
  `|x|·H_k(x)` is a **lower bound on any k-th-order code** (a code whose model depends on the last k symbols),
  and `H_{k+1} ≤ H_k`. **This is the formal target for the residual coder** — the flicker-residual is NOT iid;
  its symbols are correlated with spatial neighbors AND with the co-located flip in the previous frame. A code
  that conditions on that context can only reach `|x|·H_k`; our current per-pair `margin_conditional_residual`
  is a low-order model that leaves the temporal correlation on the table (see §4). The scribe note flags the
  `H_k(aaaa…)=0` degeneracy and Manzini's **modified `H_k*`** fix (`H_0*(x)=(1+log|x|)/|x|` when `H_0=0`) so
  the bound counts the length of near-constant runs — directly relevant: long temporally-constant flip runs
  should cost ~`log(run)`, not `run·ε`.

- **Burrows-Wheeler Transform (BWT) — "aspires to give H_k for ALL k" (this IS bzip2):** append `$`, sort all
  n rotations, output the first column. Clusters `successors(w)` together (their `banana → nn$aaab`), so equal
  contexts become runs — **without ever specifying k or a context model.** Invertible in `O(n)` (LF-mapping /
  last-first hashing). Constructible from the **suffix array** (§3, L10/L11): BWT = the characters preceding the
  sorted suffixes.

- **MTF transform + RLE + the two Manzini bounds (Thms 4–5, Manzini JACM 2001 [2]):**
  - `BWT + MTF + arithmetic ≤ 8·|T|·H_k(T) + (1/4)·|T| + O(Σ^{k+1} log Σ)  ∀k`
  - `BWT + MTF + RLE + arithmetic ≤ 5·|T|·H_k*(T) + f(k)  ∀k`
  **This is the free-lunch property:** the pipeline reaches (a constant times) the k-th-order entropy for ALL k
  simultaneously, with NO hand-designed context model. The `+RLE` version drops the additive `(1/4)|T|` term
  (which would be `~0.25 B` per residual symbol — non-trivial at ~276K flips). For a signal whose optimal
  context order we do NOT know a priori (spatial? 1-frame temporal? multi-frame?), BWT+MTF+RLE+arithmetic is the
  strongest zero-tuning baseline.

**References the note cites (all read/located):** Huffman IRE 1952 [1]; Manzini JACM 48(3):407–430 2001 [2]
(the two BWT bounds); Moffat-Neal-Witten TOIS 1998 [3] (integer/low-precision arithmetic coding — the
implementable form); Witten-Neal-Cleary CACM 1987 [4] (the canonical arithmetic coder).

## 2. THE SUCCINCT PRIMITIVES — L12 (the flip-POSITION sets; already at optimum in-tree)

- **"Succinct" = information-theoretic-optimum space + lower-order term** (get the right *constant* on the
  leading term, not just `O(·)`).
- **rank/select (Jacobson [J89] two-level; the workhorse):** `rank(i)` = #1s at-or-before `i`; `select(i)` =
  index of the i-th 1. **`n + o(n)` bits, `rank` in O(1)** via superblocks of `(lg n)^2` (cumulative ranks cost
  `O(n/lg n)` bits) + sub-blocks of `½lg n` (local counts `O(n·lglg n/lg n)` bits) + a bottom lookup table.
  `select` is harder (Clark-Munro [CM96]) but O(1) achievable. **We do not need runtime rank/select for a KB
  payload; the relevance is the SPACE bound as the target for the flip-position bitmap.**
- **A set of `k` flip-positions among `N` annulus pixels** costs `lg C(N,k)` bits at the information optimum —
  and **we ALREADY hit this**: `encode_combination_colex` (L31 colex rank) in `src/tac/codec/pr101_polymorphic.py`
  serializes the position subset as its combinatorial rank among `C(N,k)` combinations = the exact succinct
  optimum, ~3 bytes in the PR101 case. So **the position-set half of Lever-D is already at the L12 floor**; the
  headroom is in the CLASS-residual symbols + their temporal structure (§4), not the positions.
- **Tree succinctness (level-order 2n+1 bits; balanced parentheses 2n+o(n); LOUDS):** `lg C_n = 2n+o(n)`
  (Catalan). NOT-RELEVANT to us — we ship no tree in the payload. (Filed for completeness; false friend for the
  coder.)

## 3. SUFFIX STRUCTURES + RMQ/LCA — L10/L11 (the BWT *construction* machinery; the index is a false friend)

- **Suffix array + LCP** (L11 §3.1): the practical linear-space object; a pattern search is `O(|P|+lg|T|)`.
  **This is how you BUILD the BWT** (BWT = preceding chars of the sorted suffixes; sorted-rotations ≡ suffix
  sort of the `$`-terminated string). So L10/L11 are the *construction* tool for the L13 BWT, nothing more for
  us. Linear-time construction (Farach-Colton et al. [3,4]) exists but off-the-shelf `bzip2` hides all of it.
- **RMQ → LCA in O(n) space, O(1) query** (Bender-Farach-Colton; Cartesian tree; ±1-RMQ; `2n`-instantiation
  table): elegant, but its use here (longest-repeated-substring, document retrieval, k-mismatch) is
  **INDEXING/searching, not compression.** NOT-RELEVANT to shrinking `archive.zip`.
- **The FM-index trap (modern, §5):** FM-index = BWT + rank/select = a *searchable self-index* in `nH_k+o(n)`
  space (Ferragina-Manzini 2000). It is tempting because it names "BWT + rank/select + `H_k`" — but we do NOT
  need to *search* the payload; we need to *compress + inverse-transform* it. Use the **BWT transform** (L13),
  NOT the FM-index. Building an FM-index would pay for `select`/locate structures we never query. Watch-item at
  most (see §7).

## 4. MAP TO THE RATE AXIS — the three live targets

**Target 1 — Lever-D flicker-residual coder (#279, LIVE).** The regional flip-residual (which boundary
segments flip per frame + their GT class) has ~250 KB spatial-regional-context entropy (operator estimate). At
the converged ep2236 base the crude coder codes ~460 flips/pair × 600 ≈ **276K flips at `b ≈ 0.99 B/flip`**
(`margin_conditional_residual`; `lever_d_nuanced_fullstack_20260612`). 0.99 B/flip × 276K ≈ 273 KB ≈ the entropy
estimate → the current coder is **already near the SPATIAL regional entropy**, but it is **per-pair** and does
NOT exploit the **temporal** axis (a lane edge that flips in frame t overwhelmingly flips at the same annulus
location in t±1). The 6.897 tools that shrink `b` toward the JOINT (spatial+temporal) conditional entropy:

  1. **Higher-order-entropy CONTEXT arithmetic coder (L13 `H_k` + our RangeEncoder).** Condition each flip's
     class-residual symbol on a context = `(spatial margin decile, local neighbor-flip states, co-located
     flip state in the previous 1–2 frames)`. This is a k-th-order model; it reaches `|x|·H_k`. Build it on the
     in-tree `RangeEncoder`/`RangeDecoder` (or `constriction`). **This is the recommended primary coder** because
     the temporal correlation is a KNOWN, named structure we can model explicitly. ESTIMATE: adding the 1-frame
     temporal context to the current spatial model plausibly buys ~10–15% on `b` (→ ~0.84–0.89 B/flip); this is
     BYTE-REASONED, not measured — #279 must byte-close it.
  2. **BWT + MTF + RLE + arithmetic (L13 Thm 5; the bzip2 free-lunch).** Stack the per-frame residual symbol
     streams **frame-major** (so temporally-constant flips become long runs), BWT+MTF+RLE+range-code the stack.
     Manzini Thm 5 guarantees `≤ 5·|T|·H_k*(T) + f(k)` for ALL k — it captures whatever temporal+spatial order
     is present **without our having to pick k**. Use it as the **zero-tuning A/B baseline** against coder #1;
     if the hand-modeled context is incomplete, the free-lunch pipeline can beat it; if #1 is well-specified, it
     wins by the constant factor. The `+RLE` is what removes the `(1/4)|T|` additive term (Thm 4 vs Thm 5) —
     ~0.25 B/flip that matters at 276K flips.
  3. **Position sets stay at the L12/L31 colex-succinct optimum** (`encode_combination_colex`, already in-tree,
     ~lg C(N,k) bits). No change needed; do NOT re-encode positions as a raw bitmap.

**Target 2 — the payload (lane-band coords/frame, pose sidecar, structured-init).** Same L13 toolkit:
temporal-delta (already L25) + range coding at the per-parameter empirical distribution. These are already
near-entropy in-tree (rate index R12/R21); 6.897 adds nothing new here beyond confirming the arithmetic-coding
choice over Huffman. LOW EV (the payload is already ~KB near-entropy).

**Target 3 — inflate runtime (#214, <30 min bit-exact).** Cache-oblivious structures (L14–L18) are
**NOT-RELEVANT / DOMINATED**: the decode bottleneck is the FREE generator compute (coord-INR forward pass,
se3/screw warp, eikonal-SDF), not cache misses over a KB structure. BWT-inverse and range-decode are both
trivially `O(n)` and finish in milliseconds on a ≤300 KB stream — no memory-hierarchy engineering needed. Close
the door.

## 5. OSS + LICENSE + inflate.py PORTABILITY

| Structure | Canonical OSS | License | Portable to our inflate? |
|---|---|---|---|
| **Arithmetic/range coder** | **in-tree** `src/tac/lossless/range_coder.py` + **`constriction`** (already a dep) | ours / constriction MIT-or-Apache-ish | **YES — use this.** Deterministic, integer, bit-exact, rule-118-clean (generic algorithm = FREE in inflate.py; only the coded video-derived payload counts) |
| **BWT+MTF+RLE (bzip2)** | Python `bz2` (stdlib, one call); Rust **`bzip2` 0.6.0** (now 100% Rust via `libbz2-rs-sys`, Trifecta) | stdlib / bzip2 (BSD-like) | **YES.** `bz2.compress`/`decompress` is deterministic + in the Python stdlib (zero new dep) for the A/B; Rust `bzip2` for a `runtime-rs/` port. Both fast, both fit the 30-min budget trivially |
| **Succinct rank/select / colex positions** | **in-tree** `encode_combination_colex` (L31); Rust `succinct` crate (rank/select), `bsuccinct-rs` (beling) | ours / `succinct` = **MIT OR Apache-2.0** | **YES for positions (already in-tree).** The Rust `succinct` crate is permissive + portable if we ever want runtime rank/select (we don't, for a KB payload) |
| **FM-index (searchable self-index)** | Rust **`fm-index`** = **MIT OR Apache-2.0**; C++ **sdsl-lite** | fm-index MIT/Apache ✓; **sdsl-lite = GPLv3 (C++)** | **Portable but UNNEEDED** (we don't search the payload). **AVOID sdsl-lite** — GPLv3 is contaminating for a shippable inflate runtime, and it is C++ not Rust/Python. If ever needed, the Rust `fm-index` (MIT/Apache) is the clean choice |

**Drop-in for the Lever-D build:** primary = in-tree `RangeEncoder`/`RangeDecoder` with a temporal-spatial
context model; A/B baseline = Python stdlib `bz2` (BWT+MTF+RLE) on the frame-major residual stack; positions =
in-tree `encode_combination_colex`. **Zero new external dependency required** — everything is stdlib or already
in-tree. (If a native fast path is later wanted, the pure-Rust `bzip2` 0.6.0 crate is the portable option.)

## 6. THE HONEST BYTE ESTIMATE + the break-even connection (NO-FAKE)

Lever-D economics (`lever_d_nuanced_fullstack_20260612`): GO requires coded-subset effective survival
`σ_eff > σ* = b / WATERLINE`, `WATERLINE = 1.273108 B/flip`. At `b = 0.99` → `σ* = 0.778`; measured best-decile
`σ_eff ≈ 0.51` → **NO-GO**. Reactivation Path 1: `b < 0.51 × 1.273 = 0.65 B/flip` (a **~34%** cut) flips it GO
at that σ_eff.

- The operator's ~250 KB regional-context entropy ≈ **0.90 B/flip** — that is the **SPATIAL-only** conditional
  entropy. A perfect spatial coder reaches ~0.90; **0.90 > 0.65 → still NO-GO on the spatial axis alone.**
- The ONLY way `b` crosses 0.65 is if the **JOINT temporal+spatial conditional entropy** is materially below
  the spatial-only 250 KB — i.e. if inter-frame flip persistence is strong enough that BWT-RLE / a temporal
  context model collapses the residual by ≳28% beyond the spatial model. **This is plausible** (lane edges are
  highly temporally persistent) but it is **an empirical byte-measurement, not something L13 guarantees.** The
  6.897 tools give the RIGHT coder to reach that joint-entropy floor; they cannot manufacture GO if the floor
  sits above threshold.
- **Honest verdict:** the tools **SHARPEN Lever-D** (lower `b` → lower `σ*` → strictly better net-S, narrower
  NO-GO gap) and are the correct coder to run. Whether they flip NO-GO → GO is decided by ONE measurement #279
  must produce: **byte-close the frame-major residual through (a) the temporal-context RangeEncoder and (b)
  BWT+MTF+RLE, and read `b`.** If `min(b) < 0.65`, Lever-D reactivates at the converged base; if not, Lever-D
  stays a NO-GO and the d_seg belongs IN TRAINING (Lever-2/5), exactly the standing verdict — but now with `b`
  measured at its true joint-entropy floor rather than the 0.99 hand-coded value.

## 7. PER-TOPIC VERDICT (LEVER / WATCH / NOT-RELEVANT)

| 6.897 topic (lecture) | Verdict | Why |
|---|---|---|
| **Arithmetic/range coding (L13)** | **LEVER** | ~1-bit-total overhead reaches `H` on skewed residual symbols where Huffman wastes ~1 bit each; in-tree already. THE coder for Lever-D's class-residual |
| **Higher-order entropy `H_k` context model (L13)** | **LEVER** | The formal target; conditioning on temporal+spatial context is the mechanism to push `b` below the spatial-only 0.90 toward the joint-entropy floor |
| **BWT + MTF + RLE (L13, bzip2)** | **LEVER (A/B baseline)** | Manzini Thm 5: `≤5|T|H_k*+f(k)` for ALL k, zero context-tuning; `+RLE` kills the `(1/4)|T|` term; stdlib `bz2`, deterministic, 30-min-safe |
| **Succinct rank/select + colex positions (L12/L31)** | **NOT-A-LEVER (already at optimum)** | Flip-position set already coded at `lg C(N,k)` via in-tree `encode_combination_colex`; nothing left to squeeze on positions |
| **Suffix array + LCP (L10/L11)** | **WATCH (construction only)** | The tool that BUILDS the BWT; off-the-shelf `bz2` hides it. Not a coder itself |
| **RMQ / LCA / suffix trees (L10/L11)** | **NOT-RELEVANT (false friend)** | Indexing/searching, not compression — does not shrink `archive.zip` |
| **FM-index (modern BWT+rank/select self-index)** | **WATCH (false friend for coding)** | A *searchable* self-index; we compress + invert, we do not search. Use the BWT transform, not the index. Rust `fm-index` MIT/Apache if ever needed; AVOID GPLv3 sdsl-lite |
| **Cache-oblivious model / B-trees / funnelsort (L14–L18)** | **NOT-RELEVANT (dominated)** | Decode is generator-compute-bound over a KB payload; no memory-hierarchy engineering buys anything within 30 min |
| **van Emde Boas / y-fast / fusion trees (L1–L4)** | **NOT-RELEVANT (false friend)** | Dynamic integer successor/predecessor query cost `O(lglg u)`; orthogonal to static compression |
| **BST dynamic optimality / splay / Wilber (L5–L9)** | **NOT-RELEVANT (false friend + STILL OPEN)** | Query-cost competitiveness ≠ compression. Conjecture UNRESOLVED as of 2024 (tango trees `O(lglg n)`-competitive, Demaine-Harmon-Iacono-Pătrașcu; splay optimality open). **Note the MTF *list* (L5, self-organizing) ≠ MTF *transform* (L13)** — same name, different object |

## 8. FOLLOW-UPS + CRITIQUES (the modern state, read beyond 2003)

- **Compressed rank/select realizing `H_0`:** **RRR bitvectors** (Raman-Raman-Rao 2002) compress the L12
  bitvector to `nH_0 + o(n)` while keeping O(1) rank — the entropy-compressed version of Jacobson. Modern SOTA.
- **Wavelet trees / wavelet matrix** (Grossi-Gupta-Vitter 2003; Claude-Navarro 2012) — generalize rank/select
  to arbitrary alphabets at `H_0` space; the wavelet **matrix** is the faster modern layout. (The L12 note
  already gestures at Grossi/Vitter/Ferragina/Sadakane succinct-suffix-tree work.)
- **FM-index → RLFM → r-index** (Ferragina-Manzini 2000 → Mäkinen-Navarro → Gagie-Navarro-Prezza 2018): the
  **r-index** is `O(r)` space where `r` = #BWT-runs — SOTA for **highly-repetitive** data. Named here because our
  temporally-persistent flicker-residual IS highly repetitive (few BWT runs after frame-major stacking) — this
  is the theoretical reason BWT+RLE should collapse it. But r-index is a *search* index; the actionable
  takeaway is the RUN-STRUCTURE insight (few runs → RLE wins), realized by plain BWT+RLE, not by shipping an
  r-index.
- **Dynamic optimality: STILL OPEN (2024).** Tango trees `O(lglg n·OPT)`; multi-splay; splay-tree constant
  competitiveness unproven. Confirms L5–L9 are a closed door for us on two grounds (false friend + unresolved).
- **bzip2 is pure Rust now** (crate 0.6.0 / `libbz2-rs-sys`, 2024, Trifecta Tech) — a clean, permissive,
  deterministic native BWT+MTF+RLE path for `runtime-rs/` if the Python `bz2` A/B wins and we want to harden it.

## 9. THE DELIVERABLE — Lever-D coder recommendation (feeds #279)

Build the #279 Lever-D residual sidecar as:
1. **Positions:** `encode_combination_colex` (in-tree, L31/L12 succinct optimum). Unchanged.
2. **Class-residual symbols — primary:** **temporal+spatial context arithmetic coder** on the in-tree
   `RangeEncoder`/`RangeDecoder`, context = `(margin decile, local neighbor-flip states, co-located
   previous-1–2-frame flip states)`. Reaches `|x|·H_k` for the modeled context.
3. **Class-residual symbols — A/B baseline (zero-tuning):** **BWT+MTF+RLE+range** via stdlib `bz2` on the
   **frame-major** residual stack. Manzini Thm 5 guarantees `H_k*` for all k; captures temporal runs the hand
   model might miss.
4. **Measure `b` byte-closed** for both; take `min(b)`. Feed `σ* = b/1.273108` into the Lever-D GO/NO-GO. This
   is the ONE decisive measurement: if `min(b) < 0.65` Lever-D reactivates at the converged base; else the
   d_seg stays IN TRAINING (standing verdict), now with `b` measured at its true joint-entropy floor.
5. **Runtime:** all decoders are deterministic, integer, `O(n)`, generic-algorithm (rule-118 FREE in inflate.py;
   only the coded video-derived bytes count) — trivially inside the 30-min budget. No cache-oblivious structure.

**6-hook wire-in:** #1 sensitivity-map ACTIVE (the `b`-vs-context-order curve is the residual-coder prior). #2
Pareto ACTIVE (lowers the Lever-D rate-axis point; the GO plane is `σ_eff > b/WATERLINE`). #3 bit-allocator
ACTIVE (temporal-context arithmetic + colex positions is the canonical residual allocator). #4 cathedral N/A
(research). #5 continual-learning ACTIVE (this toolkit + the `b`-floor reasoning). #6 probe-disambiguator ACTIVE
(the two-coder A/B disambiguates "hand-modeled context" vs "BWT free-lunch"). **Mission:** `frontier_breaking_enabler`
(a $0 toolkit + concrete coder that minimizes Lever-D's counted bytes and decides its reactivation). **Pointer
UNMOVED 0.19110.** No score asserted, no GPU, no paid spend, no MPS.

## 10. Cross-references

`lever_d_nuanced_fullstack_20260612` (the σ*=b/WATERLINE economics + NO-GO + reactivation Path 1 `b<0.65`) ·
`lever_d_margin_conditional_residual_coder_20260610` (the current 0.99 B/flip coder) ·
`canonical_research_index_rate_20260629` (R12 finishing-kit / L30 range / L31 colex / L25 temporal-delta;
rate de-risked-cheap, d_seg binds) · `src/tac/lossless/range_coder.py` (in-tree arithmetic coder) ·
`src/tac/codec/pr101_polymorphic.py` (`encode_combination_colex`, `encode_huff_length_rank`) · task #279 (the
Lever-D build this feeds). Manzini JACM 48(3):407–430 2001 (the two BWT bounds); Witten-Neal-Cleary CACM 1987
(arithmetic coding); Jacobson PhD 1989 + Raman-Raman-Rao 2002 (rank/select, RRR); Ferragina-Manzini 2000 +
Gagie-Navarro-Prezza 2018 (FM/r-index). means≠ends; pointer UNMOVED 0.19110.
