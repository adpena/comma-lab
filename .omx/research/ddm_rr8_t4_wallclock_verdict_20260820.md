# ddm_rr8 T4 wall-clock row — THE DECODE WALL IS CLEARED (464.6 s vs 1,419.9 s), score bit-identical

`date_utc: 2026-08-20` · `owner: MAIN` · `axis: [contest-CUDA T4, n600]` · `call_id:
fc-01M0FZKTSY9ZRH2TEX27TZACKP` · `score_claim: true (identity)` · `frontier_moved: false (by
construction — archive bytes unchanged)` · cost ≈ $0.16

## THE ANSWER, FIRST

The FreeCorrector native port **clears the shipping decode wall with margin at every corner.**
Measured inflate on the shipping axis: **464.558564563 s**, against jg5's 1,419.9042126240001 s —
a **3.056× end-to-end speedup, saving 955.3 s** — and the score is **bit-identical**.

| Quantity | jg5 (Python corrector) | rr8 (NativeFreeCorrector) | verdict |
|---|---|---|---|
| final score | 0.14839100138338618 | **0.14839100138338618** | IDENTICAL |
| d_seg | 0.00020139 | 0.00020139 | IDENTICAL |
| d_pose | 6.37e-06 | 6.37e-06 | IDENTICAL |
| archive bytes | 180,625 | 180,625 | unchanged (same archive sha `f3bce5d2…`) |
| **inflate wall** | **1,419.904** s | **464.559** s | **−955.346 s** |
| evaluate wall | 51.4 s | 39.685 s | — |

Score identity is FORCED here (identical archive bytes, decode proven bit-identical by rr6), so it
is a **control that passed**, not a result. The result is the wall-clock.

**Port activation is proven in the receipt, not asserted:** `token_decoder.free_corrector =
"NativeFreeCorrector"` on this row vs `None` on jg5. A silent fallback to the Python path would
have shown `None` and the timing would not have moved. It did both.

## THE DECISION — against the MEASURED job wall, not the projected band

**CORRECTION (self-audit, same day).** The first version of this section decided against wc2's
`[822, 1302] s` "CI residual window" as though it were a measured budget. **It is not.** wc2 labels
it at source: *"`T_residual(CUDA)` for inflate | [822, 1302] s | ua2:189 (PROJECTION; step seconds
estimated)"* (`ddm_wc2_wall_clock_pass_20260820.md:273`, and again at :86 — *"ua2's per-step seconds
are ESTIMATED"*). Its payload sizes are measured; its seconds are not. Deciding a shipping question
on a projected threshold while calling it measured is the [[measured object vs named object]] genus
in my own verdict. The arithmetic below is re-stated against the one quantity that IS measured at
source — the **1,800 s whole-job wall** (`upstream/.github/workflows/eval.yml:30`,
`timeout-minutes: 30`, per #835's "budget surface, not a wall").

| | seconds | class |
|---|---:|---|
| measured inflate | **464.559** | MEASURED (this row) |
| measured evaluate | **39.685** | MEASURED (this row) |
| **our measured total** | **504.24** | **MEASURED** |
| job wall | **1,800** | MEASURED (`eval.yml:30`) |
| **left for checkout/deps/download** | **1,295.8** | derived from measured |
| what those terms are projected to cost | 498 – 978 | **PROJECTION** (wc2/ua2) |
| projected whole job | **1,002.2 – 1,482.2** | mixed |
| **slack vs the 1,800 s wall** | **317.8 – 797.8 s** | fits at BOTH ends |

Under the pessimistic host assumption (next container as slow as jg5's, ×1.2855 — itself a
one-sample estimator, see the decomposition below) our measured total re-scales to 648.2 s, and the
projected whole job to 1,146.2 – 1,626.2 s: **still inside 1,800 s at both ends.**

**jg5 for contrast: 1,471.33 s measured + 498–978 projected = 1,969.3 – 2,449.3 s — over the job
wall at BOTH ends of the projection, not merely the loose one.** That is a stronger statement than
the version this corrects, and it does not depend on the projected band being right: jg5 exceeds
1,800 s the moment the other terms cost anything at all above 328.7 s.

**The port ships** — and the residual uncertainty now sits explicitly in the PROJECTED terms
(checkout/deps/download), not in our measurement.

## HONEST DECOMPOSITION — ~13.7 s of the saving is NOT the port

The non-token stages also got faster, and the port does not touch them:

| stage | jg5 | rr8 | ratio |
|---|---|---|---|
| token_decode_or_checkpoint_load | 1,341.540 | 403.698 | 3.323× |
| neural_render_and_resize | 54.536 | 42.425 | **1.285×** |
| frame0_selector_and_io | 5.145 | 3.668 | **1.403×** |
| archive_setup | 0.358 | 0.280 | **1.279×** |

`neural_render_and_resize` is a stage the FreeCorrector cannot influence, so its 1.285× is a
**host-variance estimator**: this container was ~1.29× faster than jg5's. Consequences, stated
plainly:

- **~13.7 s** of the 955.3 s saving is a faster container, not the port.
- **Port-isolated effect** = the token stage: −937.8 s, **3.323× raw**, **2.585× host-adjusted**.
- The decision above does not depend on either ratio — it applies the host factor as a *risk
  multiplier on the absolute number* and still passes.

## cd1's LOCAL SPLIT DOES NOT TRANSFER — rv15's F4 confirmed by receipt, not argument

`ddm_cd1` measured `port_scope_seconds = 917.929` of the 1,341.5 s token stage locally, implying a
423.6 s non-corrector residual (the model forward). **The measured post-port token stage is 403.7 s
— BELOW that residual.** If cd1's split transferred numerically, the ported corrector would have to
take negative time. It does not transfer.

This is exactly what `ddm_rv15` finding **F4** warned: cd1's two halves were measured on unmatched
instruments (local torch=6 threads, BLAS unpinned) vs T4 (torch=1, OMP=MKL=1), against our own
pin-`(code, weights, threads, batch)` law. F4 is now CONFIRMED with a receipt. The cure is not to
re-derive a corrected split — it is to **stop citing cd1's absolute seconds on the T4 axis at all**;
the end-to-end row supersedes them. Another instance of the [[cross-regime constant transfer]]
genus, caught before it priced a decision.

**rv15 F2/F3 (the `2.03×/2.77×` bar published without its ±61 s band, and the k=2 corner missing by
−7.6 s) is now MOOT for the shipping decision.** A direct end-to-end measurement replaces a modeled
threshold. F2/F3's *publication* cure still stands wherever the bar is quoted historically; it is no
longer load-bearing for whether the port ships.

## RUNTIME MANIFEST — 36 vs 35 files reconciled, not a discrepancy

The seal (`CANDIDATE_SEAL_rr8_instrumented.json`) counts **36** files in
`candidate_runtime_jg5_native_corrector_instrumented/`; the eval receipt's
`inflate_runtime_manifest.runtime_file_count` is **35**. Measured cause: the eval's runtime manifest
**excludes `archive.zip`** (verified — no `archive.zip` in its file list), because the archive is
hashed separately as the scored payload. 36 − 1 = 35. The two digests are different functions over
different sets by design; neither is drift.

- eval runtime_tree_sha256: `69f36aa703576ba64161c25d832ddc8128d1b15a8843fad5b1760c5ec0fb5c8e`
- eval runtime_files_sha256: `248c52b26a49a333c6255d228f2f4a2f734ffd20c82fb23dcbecca977a1419da`
- seal content-only digest: `b8a43c6bc4ab14b65d021f576755ebc3c14857f90e149bf0aef7dd8080bce1f9`
- SHIPPED tree (jg5 pointer): `2103073d739fc3f27d329ea0785ea3010307360c2380af0476e16d0f5b57cb9b`

The instrumented tree is NOT the shipping tree — it carries per-stage timing emission. The shipping
candidate must be the **clean** port (instrumentation removed or proven byte-neutral), re-sealed.

## WHAT FIRES NEXT — ONE composed row, not two

Two transforms are now proven and BOTH move the runtime tree:

1. **rr8 native corrector** — archive byte-identical, runtime tree changes, decode 3.056× faster.
2. **rr5 CPR1 rider** — archive `df7fd266…` @ 180,456 B (−169 B, ΔS −1.125302e-04), runtime tree
   changes (its `inflate.py` carries the adaptive-arithmetic restore path).

They are orthogonal in mechanism and compose. Firing them separately buys a T4 row for a tree we do
not intend to ship. **The shipping candidate is ONE object: {rider archive `df7fd266…` × ported
+ rider runtime}**, sealed and fired as ONE row. Expected: S **0.14827847122030854** (exact rate
arithmetic, decode identity proven for the rider leg) at an inflate wall near 464 s.

Fire order, now that rr8 is adjudicated:
1. Compose the clean (non-instrumented) port with the rider runtime; prove decode identity on the
   composed tree.
2. Seal ONE candidate (`make_candidate_seal.py`; single-flight).
3. ONE T4 row → if it lands, that is the sixteenth pointer move AND the decode wall closed together.
4. The packet's declared runtime + the 4,369.6 s contest-CPU figure both need re-derivation against
   the composed tree — the CPU wall should fall by a similar factor, which may reopen the CPU-axis
   question the packet currently answers as MEASURED-INFEASIBLE.

## Own-vehicle frontier

**S 0.14839100138338618 @ 180,625 B [contest-CUDA T4 n600] — UNMOVED by this unit, by
construction.** This row bought a wall-clock, not a score: the archive bytes were deliberately held
identical so the port's decode-identity would be measured against a forced score control. The
control passed and the wall cleared.
