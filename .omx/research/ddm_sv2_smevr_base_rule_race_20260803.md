# ddm_sv2 — the SMEVR base-rule race, re-run on the live field: the lever is DEAD, and one mechanism kills three of them

**Axis:** `[macOS-CPU advisory]` — rate accounting only. NO scorer run, NO score claim.
Own-vehicle frontier **S = 0.7910689 @ 353,805 B UNMOVED**. Borrowed pointer 0.1910828242 UNMOVED.
`score_claim=false` · `research_only=true` · `promotable=false`.

**Task:** `#940` leg (b) — the SMEVR base-rule race (`#859`), plus `#858`.
**Answer to `#859`: REFUTED on the live vehicle.** The −2,781 B lever was real on the
DR7T field and does not survive the representation change to `IX2TOK01`. No bytes are
available. **Answer to `#858`: NOT A LIVE DEFECT** (re-derived at source, independently of
the sister adjudication).

---

## §1 What the charter asserted, and what is actually true

| charter claim | verdict | evidence |
|---|---|---|
| "measured **−2,781 B (ΔS −0.0018518)**, replicated" | **REAL but FIELD-SCOPED, and "replicated" is overstated** | receipt `ddm_rh1_token_rate_decomposition_20260801.json` `base_race.v4d_shipped[0]`: `byte_delta -2781`, `rate_delta_s -0.0018517537486327585`, `rule "alpha=0.0,p=2.0"`, `cells_moved 774`. The second field gave **−1,172 B at p=1.5**, not −2,781. What replicated is the SIGN, not the magnitude or the optimal exponent. |
| "blocked on a receiver/format change" — unblocked by sub1 | **the blocker was correctly identified but is NOT what unblocks it** | `ddm_r7_token_coder.py:1505` — the `canonical` verify rung re-encodes via `factor_mode_delta` and refuses a non-mode base. `decode` itself (`:1501`) is base-agnostic. So it was never a container/grammar change; sub1's chain was not the gate. |
| "SMEVR everywhere optimal" | **REFUTED BY RACE** | SMEVR loses to the shipped arrangement by **+5,183 B** on the live field (§4). |
| `#858` receiver admits an ambiguous absent `token_codec` | **NOT A LIVE DEFECT** | §5 |

**The decisive fact the charter missed:** `qd2` (`ddm_qd2_rebaseline_against_cx1_20260803.md:382-384`)
had already flagged the −0.0018518 figure **NOT-COMPARABLE** — "the token representation changed
underneath it" — and asked for exactly this $0 re-measurement. This memo is that re-measurement.

## §2 The live field is not the field the lever was won on

The live frontier archive (`ddm_pu2_20260803/submission_pu2/archive.zip`, sha `c72ef357`,
**353,805 B**) reproduces byte-identically through `tac.submission_chain.build_byte_ledger`
(residual 0, payload re-encodes identically). Its bulk is **341,295 B** and its magic is
**`IX2TOK01`** — *not* a `DR7T` SMEVR frame. **SMEVR is not live on the current vehicle.**

`ddm_iv1_repack_rungs.py:19` names the transition: `DR7T` → `IX2TOK01` is a **coder generation
change**, not a content change. The generation moved two things at once:
1. residual stored **cell-major** `(R,C,K,P)` — every cell's 600 temporal values contiguous;
2. the coder chosen by a **4-way race** (`code_block`: stored / zlib / brotli-q11 / raw-LZMA1).

## §3 The exact rh1 family, re-raced on `IX2TOK01` — every arm loses

Family re-implemented verbatim from `ddm_rh1_rate_harvest_20260801.md:§5`:

```
base(cell) = argmin_b  Σ_s hist[cell,s] · C((s − b) mod L)
C(r)       = 0 if r == 0 else α + circdist(r)^p
```

**GAUGE (run first, and it must pass or nothing below is measuring the real object):** the
incumbent mode base re-encodes the shipped bulk **byte-identically** — 341,295 B = residual
339,970 + base 1,297. Every arm below asserts full lattice decode-equality BEFORE its bytes
are counted, so a lossy arm cannot post a byte win.

30 arms, α ∈ {0, 0.5, 1, 2, 4} × p ∈ {0.5, 1, 1.5, 2, 2.5, 3}:

| α | p | frame B | vs shipped | cells_moved |
|---:|---:|---:|---:|---:|
| — | — (**incumbent mode**) | **341,295** | **+0** | 0 |
| 4.0 | 0.5 | 341,585 | +290 | 186 |
| 0.5 | 0.5 | 341,605 | +310 | 391 |
| 0.0 | 0.5 | 341,606 | +311 | 471 |
| **0.0** | **2.0** ← *rh1's v4d winner* | **341,932** | **+637** | **774** |
| 0.0 | 2.5 | 342,308 | +1,013 | 811 |
| 0.5 | 2.5 | 342,381 | +1,086 | 799 |

**The rule is faithfully reproduced; only the verdict flipped.** rh1's winning arm moves
**774 cells** here — byte-for-byte the `cells_moved: 774` in rh1's own receipt. Same rule, same
cells, opposite sign. That is the strongest available evidence that this is a genuine
field-scoped sign flip and not a re-implementation error.

**On the live field the incumbent mode is the MINIMUM of the entire rh1 family.**
This directly overturns rh1's structural claim that "the incumbent mode is the WORST corner of
the family on both fields" — true on both DR7T fields, **false on `IX2TOK01`**.

## §4 SMEVR raced, not presumed — it LOSES by 5,183 B

Same lattice, four arrangements, all lossless-verified (`VERIFY_CANONICAL`):

| arm | bytes | vs shipped |
|---|---:|---:|
| **shipped `IX2TOK01`** (mode + cell-major + coder race) | **341,295** | **+0** |
| SMEVR native `(P,R,C,K)` — the DR7T generation | 346,478 | **+5,183** |
| SMEVR cell-major | 359,626 | +18,331 |
| brotli11 native (r7) | 396,442 | +55,147 |

SMEVR native lands at **exactly 346,478 B**, the v4d shipped token size
(`ddm_r7_smevr_liveness_on_v4d_20260801.json`), confirming the lattice content is identical and
the cross is apples-to-apples.

Feeding SMEVR the cell-major layout makes it **worse by 13,148 B** than feeding it the native
layout — its adaptive contexts are built for the native temporal ordering, so the layout that
helps LZ actively breaks the context model. The two wins are not composable.

## §5 `#858` — re-derived at source, not a live defect

`ddm_tr1_runtime.py`:
- `:301` `if set(value) != expected_keys: raise` — **strict key-set equality**; an unexpected key is refused.
- `:317-319` — membership, not `is not None`: `if "token_codec" in value and value["token_codec"] != TOKEN_CODEC_R7_SMEVR: raise`. An explicit `null` is refused as *"a second spelling of the legacy framing."*

Three cases, all distinguished: **absent** → legacy Brotli-Q11 (byte-identical to the pre-codec
runtime), **present == `r7_smevr_v1`** → SMEVR, **anything else incl. `null`** → refused. Pinned by
`test_legacy_token_codec_stays_absent_and_byte_identical`. The `subagent_contract.py:411` comment
that reads like a live bug is HISTORICAL PROVENANCE of the original discovery.

## §6 The one mechanism — and what it forbids next

Four independent levers were measured and **all four lose, for the same reason**:

| lever | cost | why |
|---|---:|---|
| rh1 base family (best arm) | +290 B | trades exact zeros for smaller nonzero magnitudes |
| zigzag residual remap | +1,996 B | preserves runs of 0 but scrambles symbol adjacency |
| SMEVR native | +5,183 B | adaptive symbol coding beaten by locality + LZ |
| SMEVR cell-major | +18,331 B | layout breaks SMEVR's own contexts |

**The `IX2TOK01` generation moved the win from *symbol-rank cost* to *match structure*.**
Mode maximizes exact-zero runs, which is precisely what brotli/LZMA are paid for; any rule that
instead minimizes a magnitude cost destroys those runs. My own zigzag prediction was wrong for
exactly this reason and I am recording it as refuted, not quietly dropping it: order-0 reasoning
is the wrong lens for an LZ-coded lane.

**Corollary (the theory check that came first):** per-cell order-0 entropy is **invariant** under
a modular base shift — the residual histogram is a permutation of the value histogram. A base rule
can therefore only ever win through *cross-cell alignment* or *the base block's own cost*, never
through per-cell entropy. Mode already maximizes cross-cell alignment on symbol 0.

**FORBIDS:** any further base-rule, residual-remap, or symbol-recoding arm on `IX2TOK01` without
first showing it *increases* match structure. The remaining token-rate headroom is not in the
coding rule — the shipped lane is 339,970 B for a lattice whose content is fixed; headroom must
come from the **content** (fewer/coarser tokens), which is the `br1` unit-drop / level-drop
surface, explicitly verified there as receiver-free.

## §7 What I did NOT do, and why

- **No n600 scorer run.** There is no candidate: every arm is ≥ the shipped bytes, and the codec is
  lossless so seg/pose cannot move. Burning the scorer slot would have measured a known zero. The
  slot stays with `et1`.
- **No coder change landed.** Landing a base-rule parameter would be building a surface for a lever
  that measures negative — the built-instead-of-paid failure.
- **The `#859` blocker is left standing but re-scoped:** the `canonical` verify rung at
  `ddm_r7_token_coder.py:1505` still pins mode-base on the DR7T line. That is now a *dormant*
  blocker: it gates a lever with no live value. It should not be paid down until some field
  re-opens a positive base-rule delta.

## §8 Reproduction

```
.venv/bin/python <scratch>/sv2_rh1_family_race.py      # §3, gauge + 30 arms
.venv/bin/python <scratch>/sv2_base_rule_race.py       # §3 sister: 6 base rules x 2 remaps
```
Both open the live archive read-only and assert the gauge before any arm is counted.

**Pinned structurally by** `test_token_frame_decode_is_base_rule_agnostic` in
`src/tac/optimization/tests/test_ddm_ix2_archive_container.py` — the fact that makes any future
base-rule race a $0 encoder-side experiment with no receiver change.
