# ddm_sub1 — the byte-closed submission chain, canonicalized

**UTC** 2026-08-04 · **commit** `d6af2e1636` · **axis** `[macOS-CPU advisory]` NON-PROMOTABLE
**Pointer** 0.1910828242 `[contest-CPU]` **UNMOVED** · own-vehicle S **0.7910689 @ 353,805 B UNMOVED**.
This is **apparatus, not a score mover**, and says so.

Operator binding 2026-08-03: *"Anything that is necessary for full, bi[te] closed, archived
submission should not be ad hoc or manual or exist in temporary probe scripts or anything like
that ... it should be canonicalized for our own frontier work."*

Landed: `src/tac/submission_chain.py` (library) · `tools/submission_chain_cli.py` (CLI) ·
`src/tac/tests/test_submission_chain.py` (33 tests, every guard mutation-checked).

---

## 1. DENOMINATORS — what is canonical now, what is still probe-only

The chain has **eight** stages. **Seven are now canonical; one remains probe-only.**

| # | Stage | Before | Now |
|---|---|---|---|
| 1 | state → payload sections (`_encode_pose_warp`, `_encode_kl1_field`) | probe `ddm_pu2_pose_tail_floor_probe.py:529,542` | **STILL PROBE-ONLY** — named debt D1 |
| 2 | payload → `archive.zip` | canonical (`ddm_ix2_archive_container.build_single_member_zip`) | canonical, now also via `stage_submission` |
| 3 | stage vendored runtime tree | shell, 5 hard-coded `/Volumes` roots | `stage_submission` + `ChainPaths` (flag → env → repo default) |
| 4 | inflate | shell, hard-coded, rc unchecked by the caller | `run_inflate` — foreground, rc captured, **three** refused modes |
| 5 | evaluate | shell, hard-coded invocation | `run_upstream_evaluate` — recomputes S from parts |
| 6 | typed receipt | **did not exist** | `ChainReceipt` (`tac_submission_chain_receipt.v1`) |
| 7 | per-section byte ledger | **did not exist for this container** | `build_byte_ledger`, closes exactly or raises |
| 8 | vendored-runtime custody | **did not exist** | `audit_runtime_tree`, transitive import closure |

**Scope of every negative claim below:** searched `src/tac/`, `tools/`, `scripts/`, `experiments/`.
A recall arm opened 14 of 283 files matching `evaluate.py`, 7 of 15 `*byte_close*` tools, and both
files matching the discriminating terms `section_ledger` / `archive_receipt`. **No exhaustiveness
is claimed.**

---

## 2. FOUR REFUTATIONS of my own charter (all measured at source)

**R1 — `repair_entropy_coder_runtime_adapters.py` is NOT counted-tree dead weight.**
The charter said it "is apparently never imported by `inflate_runner_v4d.py` (:46-88)". True as
stated, and **misleading**: `ddm_r7_token_coder.py:37` imports it. It is a *second-hop* dependency;
**deleting it breaks inflate.** `audit_runtime_tree` therefore closes the import graph
**transitively**. Executed control (`test_custody_import_closure_is_transitive`, mutation-checked):
a one-level scan reaches 5 modules and **misses** the adapter; the transitive closure reaches 6.
**Unreached files on the shipped tree: 0.** There is no dead weight to remove.

**R2 — the live archive is ONE ZIP member, not two.** Measured: `0.bin` only.

**R3 — per-section byte ledgers already exist.** The charter said "NO standalone byte-size ledger
exists anywhere". Two do:
`tac.optimization.ddm_tr1_runtime.section_ledger` (TR1 packet-directory format) and
`tac.boundary_math.integer_plane_emitter_byte_close.archive_receipt` (C2 integer-plane archive).
Neither parses the **ix2 two-tier payload** the frontier actually ships, which is the scoped reason
a third exists. Per #533 the module **discloses both siblings in its docstring and adopts the C2
sibling's field names** so the three agree on vocabulary.

**R4 — the shipped `inflate.sh` has no repo source of truth.** It matches **none of 76** tracked
`inflate.sh` files. It still invokes a bare `python`, which is the shipping hazard `ddm_si1`
measured (rc=127 on a python3-only host). si1 fixed the *emitters*; **the tree our live frontier
actually ships was not re-staged and still carries the old form.** Debt D2.

---

## 3. TWO VACUITY BUGS FOUND IN MY OWN GUARDS

Both found by **mutation testing**, not by review. Both are the disease this arm exists to cure,
committed by the arm curing it.

**V1 — the ledger closure could never fire.** The first draft back-computed ZIP framing by
subtraction, making `residual == 0` **an algebraic identity**. Mutation 3 (`closes()` → `return
True`) left the suite **green**, which is how it surfaced. Cured: framing is **predicted
independently** by the container's own `zip_framing_overhead` and compared against the measured
remainder, **and** the payload is re-encoded and compared byte-for-byte. Two independent legs,
neither derivable from the other. Control: a 64-byte ZIP comment now makes it refuse; reverting to
subtraction turns that test RED.

**V2 — the deflate test carried an escape clause.** `assert not closes() or compress_type != 0`
passed trivially while the ledger **silently accepted** a DEFLATE repack. Cured: non-STORED members
are refused (the canonical packer emits STORED only, and the framing predictor models STORED).
Mutation-checked.

**Law:** a guard written and tested in the same sitting is tested against the author's model of it.
Mutation is the only thing that tests it against the code.

---

## 4. ACCEPTANCE TEST — the byte-identity control PASSES

Against the live own-vehicle frontier (`ddm_pu2`, sha `c72ef357…`, 353,805 B):

```
byte-identical      : True  (sha + size)
ledger closes       : True  (residual 0; predicted framing 108 == measured 108)
payload re-encodes  : True  (canonical encoder reproduces the shipped payload byte-for-byte)
```

**The standing byte ledger** — the number no memo has to re-derive by hand again:

| part | bytes | share |
|---|---:|---:|
| ZIP framing (1 member `0.bin`) | 108 | 0.03% |
| payload header | 5 | — |
| **bulk (tokens, STORED)** | **341,295** | **96.46%** |
| joint count byte | 1 | — |
| joint group (CODED) | 12,396 | 3.50% |
| — config (raw) | 36 | |
| — renderer (raw, `IX2REN01`) | 3,266 | |
| — selector (raw, JSON) | 535 | |
| — pose_warp (raw, `PFS1WPD1`) | 8,751 | |
| joint raw total / coder saving | 12,588 / **−192** | |
| **archive total** | **353,805** | **residual 0** |

Raw section size ≠ rate cost: the joint group's **counted** cost is 12,396 B, not the 12,588 B raw
sum. The ledger keeps the two named apart so neither can be quoted as the other.

**The inflate leg, exercised END-TO-END on the real archive through the canonical chain** (not a
unit fixture):

```
rc          : 0            (job's own rc, recorded separately from the launcher's per si1)
raw files   : 1  (0.raw)
raw bytes   : 3,662,409,600
seconds     : 200.5
```

3,662,409,600 B = 1200 frames × 874 × 1164 × 3 **exactly** — the decode is structurally correct for
600 pairs, not merely non-empty. This closes the gap that the unit controls alone could not: a guard
that has only ever run on a fixture has not been shown to run on the real thing.

**Vendored-runtime custody on the shipped tree:** 3 IDENTICAL · **3 DIVERGED** · 1 UNMAPPED.
`ddm_r7_token_coder.py` (549 lines differ), `ddm_tr1_runtime.py` (262), and
`repair_entropy_coder_runtime_adapters.py` (15) all differ from HEAD — HEAD has moved forward
(R7 codec support, verify ladder) and the vendoring applies a flatten transform
(`from tac.optimization.X` → `from X`). **The shipped receiver is a PINNED copy; re-staging from
HEAD today would ship a different receiver.** That is now recorded on every receipt instead of
being invisible.

---

## 5. WHAT I DID **NOT** DO

- **Did not re-run `upstream/evaluate.py`.** `ddm_si1` already independently re-verified this exact
  row (components recompute to S=0.7910689, delta 1.5e-9, over a stated n600 denominator; archive
  sha matches; independently re-inflated bit-identically). Spending an authorized n600 slot to
  re-derive a number a sister arm verified today is redundant. The scorer slot is **RETURNED UNSPENT**.
- **Did not modify** `experiments/ddm_pu2_pose_tail_floor_probe.py` (read-only per charter), nor the
  three protected artifacts.
- **Did not delete the probe scripts.** They are HISTORY — the record of how the row was produced.
  The CLI docstring names the supersession explicitly.
- **Did not duplicate si1's inflate fix.** Consumed it: the module never backgrounds, captures rc,
  and additionally refuses rc=0-with-no-output.

---

## 6. DEBT — every row OWNED with a fire-condition

| id | debt | owner | fire-condition |
|---|---|---|---|
| **D1** | Stage 1 (section encoders `_encode_pose_warp` / `_encode_kl1_field`) is still probe-only. They are the **inverse of the shipped receiver's parser** and belong beside it in `tac`. | ddm_sub1 successor | fires the next time any arm needs to rebuild a pose_warp section — i.e. the next pose-axis row |
| **D2** | Shipped `inflate.sh` matches no tracked file and carries the bare-`python` hazard. | si1 successor / next submission stager | fires before ANY real submission, and immediately if a contest runtime is python3-only |
| **D3** | 3 vendored runtime files diverge from HEAD; the divergence is recorded but not adjudicated (pin forward, or re-stage and re-verify the decode). | next byte-close arm | fires when a row needs a receiver feature only HEAD has (e.g. R7 token codec) |
| **D4** | `run_upstream_evaluate` has unit controls but has **never run on the real 600-pair archive through this module**. | next arm that needs an exact row | fires on the next authorized n600 eval |
| **D5** | `tools/levelset_byte_close_and_eval.py` still carries its own private copy of the evaluate wrapper; it should import the canonical one. | next levelset arm | fires when either copy changes |

---

## 7. WHAT I REFUTE IN THIS CHARTER — summary

Four of the charter's factual premises were wrong (§2), and it asked me to treat a live transitive
dependency as deletable dead weight. The charter's *instinct* was right — the chain was ad hoc — but
its **inventory** was not, and following it literally would have shipped a broken archive. The
charter also asked for a guard-with-a-control per property; applying that standard **to my own work**
is what found V1 and V2.
