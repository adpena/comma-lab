# ddm_to1 — the tail override: ma1's within-miss law on the live ck2 pointer body

**Arm** `ddm_to1_tail_override_20260819` · **axis** `[macOS-CPU exact byte/container]`
· **score_claim** false · **promotable** false · **twelfth move**

STORES CONSULTED: `.omx/state/canonical_frontier_pointer.json` (live pointer re-read
at build) · `/Volumes/APDataStore/pact/ddm_ck2/{seal,compile/build_r1,advisory,generations}`
· `/Volumes/APDataStore/pact/ddm_ma1/MA1_RETENTION_MANIFEST.json` ·
`/Volumes/APDataStore/pact/ddm_rc1x/{RESULT_rc1x_e2e_verify.json,ma1_byteclose_driver.sh,byteclose_ma1}`
· `/Volumes/APDataStore/pact/ddm_fx2/byteclose_a` · `experiments/ddm_sa3_rebase_sz1.py`
· `experiments/ddm_ma1_within_miss_corrector.py` · CLAUDE.md non-negotiables.

## The headline

The ck2 pointer body now carries ma1's within-miss law. **176,420 B**, sha
`50e561454b23026d3870f056747e848a49bd5f2b1e23930155d1281aeee91927`, −105 B against
the live pointer, **net ΔS = −6.991519e-05**, a pure rate move.

## What was structurally missing

`ddm_sa3_rebase_sz1.build_candidate` re-serialises the semantic and carrier sections
and then **borrows sz1's token tail verbatim**:

    member = HEADER + sz1["hpac"] + semantic_stream + carrier_stream + sz1["tail"]

There has never been a build step that substitutes a *re-encoded* tail. So every rate
win measured on the token stream — ma1's included — was unreachable from the pointer
body, no matter how well it was measured elsewhere. That missing step is this arm.

## Why the substitution is IDENTITY, not transfer

`ddm_ma1` measured −104.584 B of code length and pre-registered −105 B on the archive,
on both the ck1 and ck2 bodies, but could not byte-close it (its blocker was the
`rc64_source` pin). `ddm_rc1x` cleared that with the two-role rc64 recipe and closed the
law end-to-end on the rr4/D1 lineage.

The question this arm had to answer is whether ma1's tail re-encodes *the object the ck2
pointer actually ships*. It does, and the proof is byte equality rather than argument:

| object | bytes | sha256 (16) |
|---|---:|---|
| fx2 D1 tail | 109,897 | `59cc27c907d645c0` |
| **ck2 tail** | **109,897** | **`59cc27c907d645c0`** |
| ma1 tail | 109,792 | `4bc30d3f8ec1aecb` |

ck2's borrowed tail **is** fx2's D1 tail, byte for byte. ma1's tail is that exact object
re-encoded under the within-miss law, −105 B. This is the `ddm_sa3` identity-gate
pattern applied to the token section, not the qs4 cross-lattice transfer disaster.

The model-side state that drives the token decode is byte-identical on both sides, and
this is measured, not inferred — from ck2's own retained decode checkpoint and rc1x's
parse-back:

| | ck2 pointer | rc1x ma1 |
|---|---|---|
| hpac_blob | `e8c0cfd7…` | `e8c0cfd7…` |
| residual_payload | `74775aab…` | `74775aab…` |
| corrected_quantized_logit | `562ac652…` | `562ac652…` |
| corrected_cdf_input | `dd48843b…` | `dd48843b…` |
| token_stream | `5b09fd78…` (fx2) | `15054e5d…` (ma1) |
| **decoded token field** | **`9ba2e52b…`** | **`9ba2e52b…`** |

Only the token stream differs. Both decode to the same field. The re-encode is rate-only
on each side independently.

One more piece of evidence is already on the paid axis: ck2 ships fx2's tail against a
*different* semantic section and scored 0.1566645 on T4. That is empirical proof the
token decode does not depend on the semantic section — the thing a splice would most
plausibly break.

## Why a tail splice is exactly a composer re-run

The RX1M member is `HEADER(14) + hpac + semantic + carrier + tail`, and the header stores
`hb`/`sb`/`cb` only — the tail is implicit (`sections()`: `outer[offset:]`). Substituting
the tail therefore moves no other field, and the build asserts that by re-parsing the
assembled archive and comparing every non-tail field against the pointer's.

## Identity controls (both mandatory, both PASS)

**(a) override OFF** — splicing ck2's own tail through the same code path reproduces the
pointer archive `0aa1cada…` @ 176,525 B **byte-identically**. This is what makes the
ON variant's −105 B attributable to the tail alone.

**(b) override ON** — 176,420 B, `50e56145…`, double-compile byte-identical; hpac,
semantic, carrier and the reserved flag all byte-identical to the pointer.

## The arithmetic

    delta_bytes            = -105
    dS_rate  = 25 * -105 / 37,545,489  = -6.991519e-05     MEASURED (exact byte count)
    dS_seg                             =  0                d_seg UNCHANGED 0.00030309
    dS_pose                            =  0                d_pose UNCHANGED 7.77e-06
    net dS                             = -6.991519e-05     pure rate

    admit bar  -3.5e-06                      ->  19.98x
    SUMMED two-row report-8dp bound           ->  10.48x

The bound is the conservative price and it is **summed over two rows, not one** — bounds
ADD for deltas, per the round-12 F1 correction. Per row: seg `100 × 0.5e-8 = 5.000e-07`,
pose `(5/√(10·7.77e-06)) × 0.5e-8 = 2.836e-06`, total `3.336e-06`; two rows
`6.672304e-06`. That bound is the honest price *if* the distortion legs were differenced
from two independently-reported 8dp receipts. Once the decode proves byte-identity the
legs cancel exactly and the net is pure rate with no distortion uncertainty at all — but
the bound is what stands until the decode lands, so it is the number quoted.

Projected S, rate-only: **0.15659459685822907**.

## The receiver carries the law

ma1 is a probability-MODEL change, so the decoder must carry it. `residual_archive.py`
does `from .free_corrector import FreeCorrector`, and ma1 exports exactly that drop-in,
so the swap is: keep ck2's fx2 as `runtime/fx2_model_axis_corrector.py` (ma1 subclasses
it and inherits its frozen `SHIPPED_CONFIG`) and stage ma1 as `runtime/free_corrector.py`.

Verified on the staged tree with the repo *off* `sys.path`: MRO is
`FreeCorrector → Ma1WithinMissCorrector → Fx2ModelAxisMixer → FixedPointLogisticMixer`,
it instantiates at the shipped 384×512 plane, and its `SHIPPED_CONFIG` is **dict-equal to
the encoder's** (`within_miss=True`, `miss_cell=nb3_prev1`, `miss_min_count=1`,
`miss_clamp=16.0`). Encoder and receiver agree on the model, which is the condition for
the arithmetic decode not to desync.

## Structural surprise, stated plainly

**`ddm_rc1x`'s `candidate_runtime` is not self-contained.** It staged
`ddm_ma1_within_miss_corrector.py` *verbatim* as `runtime/free_corrector.py`, leaving
`from experiments.ddm_fx2_model_axis_corrector import …` and a dynamic
`__import__("experiments.…")` in the shipped tree. Its parse-back passed only because the
driver exported `PYTHONPATH=/Users/adpena/Projects/pact`. That tree would fail at contest
decode, where no repo is on the path. This arm rewrites those to relative imports, fails
closed on each pattern, re-parses the result, and asserts no `experiments.` reference
survives. rc1x's *measurements* stand — the encode and the decoded-field identity are
unaffected — but its runtime tree is not shippable as staged.

## A defect I introduced and fixed

My first `splice_tail` asserted `member[:offset] == outer[:offset]` after building
`member = outer[:offset] + tail`. True by construction; it could never fire. A gate that
cannot fail is not a gate — the detector-zeroes-on-the-cure failure, in my own code, in
the one function that carries the whole claim. Replaced with a re-parse of the assembled
archive, which exercises the header arithmetic, the zip writer and the reader the
receiver uses; and a companion test asserts the replacement has discriminating power by
mutating a semantic byte and requiring the parse to see it.

## Decode wall-clock

ma1's token stage measured **572.36 s** (rc1x) against fx2's **571.43 s** (ck2's own
advisory) on the same macOS-arm64 CPU instrument, 600 pairs, python token decoder:
**+0.93 s, +0.16%**. Well inside budget. ck2's T4 decode was ~941 s. The local advisory
on this arm's own tree gives the definitive figure for this tree.

## The decode, measured on THIS candidate

`fire_local_advisory` attempt_0002, macOS-arm64 CPU, 600 pairs, python token decoder.
The token stage settles the whole question:

| | ck2 pointer | **to1 candidate** |
|---|---|---|
| archive | `0aa1cada…` 176,525 B | **`50e56145…` 176,420 B** |
| token_stream | `5b09fd78…` (fx2) | **`15054e5d…` (ma1)** |
| hpac_blob | `e8c0cfd7…` | `e8c0cfd7…` |
| residual_payload | `74775aab…` | `74775aab…` |
| corrected_quantized_logit | `562ac652…` | `562ac652…` |
| corrected_cdf_input | `dd48843b…` | `dd48843b…` |
| **decoded token field** | **`9ba2e52b…`** | **`9ba2e52b…`** |
| token stage | 571.43 s | 578.08 s |

The candidate ships a **different token stream** and decodes to the **same token field**.
That the stream differs is what makes the control non-vacuous — the override is provably
in effect, not a no-op wearing its name. Everything downstream of the token field is a
deterministic function of it and of sections this arm did not touch.

Token-stage cost: **+6.66 s, +1.17%** against the pointer on the same instrument. ck2's
T4 decode was ~941 s; this is comfortably inside the 30-minute budget.

### F5: the inflated output is byte-identical

    candidate  0.raw  3,662,409,600 B  ccbfa3327d0f2486f8a2d7970fe89c5d56302eb1e04714d05eabff52278f1f9d
    pointer    0.raw  3,662,409,600 B  ccbfa3327d0f2486f8a2d7970fe89c5d56302eb1e04714d05eabff52278f1f9d

Equal by sha256 **and** by a direct `cmp` of the two files on disk. F5 discharged.

This closes F2 and F3 at the strongest available level. The distortion legs are not
"equal to 8dp" — they are equal **by construction**, because the scorer is a
deterministic function of an output that is byte-for-byte the same. So the two-row 8dp
bound quoted above is the conservative price and no longer the binding one: the net is
pure rate and it is exact.

**The advisory is therefore unnecessary for pricing this row, and this is stated plainly
rather than hedged**: byte-identity of the decoded output is strictly stronger evidence
than a scorer comparison, which could only ever agree to the reported decimals. The
advisory was run because its dominant cost *is* the decode that produced this proof; its
scorer half is confirmatory and blocks nothing.

## What is owed

**NO Modal fire from this arm — MAIN owns T4.** The fire-order is: the seal
(`/Volumes/APDataStore/pact/ddm_to1/seal/CANDIDATE_SEAL_to1_r1.json`, sha
`33e9239738d7fd7a…`, `SEAL_VALID`), then a T4 row on
`/Volumes/APDataStore/pact/ddm_to1/generations/to1_tail_override_r1`.

An advisory is **not** needed to price this row: byte-identity of the decode is strictly
stronger than a scorer comparison, because it makes the distortion legs equal by
construction rather than equal to 8dp. The advisory was run anyway, since its dominant
cost is the decode that produces the identity proof.

## Custody

`/Volumes/APDataStore/pact/ddm_to1/` — `compile/` (both archives: the control and the
candidate, per-candidate not winners-only), `generations/to1_tail_override_r1/` (the
staged receiver), `advisory/attempt_0002/`, `TO1_RETENTION_MANIFEST.json` with sha256 +
bytes for every retained file.

## Commits

`a9fd795b0f` — the builder, the composer `tail_override` hook, 15 tests.
