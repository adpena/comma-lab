# ddm_sa3 — re-basing the compensated semantic edit onto the live pointer

Axis: **[macOS-CPU exact byte/container + receiver parse-back]** for every byte, and
**[macOS-CPU advisory, frozen CPU-torch PoseNet]** for every d_pose. `score_claim=false`,
`promotion_eligible=false`. No Modal, no lane claim, no scorer-lane fire — MAIN fires.

**Own-vehicle frontier: `S = 0.15771357797660338 @ 179,930 B [contest-CUDA T4 n600]` — UNMOVED by this arm.**

## VERDICT — the mechanism survives the re-base; the candidate is rebuilt and sealed

| | sa2 (obsolete) | **sa3 (this arm)** |
|---|---:|---:|
| base | rr4 0.158533 @ 181,161 B | **sz1 0.15771357797660338 @ 179,930 B** |
| candidate bytes | 179,851 | **179,140** |
| Δbytes vs its own base | −1,310 | **−790** |
| projected S | 0.157896 (**worse than the live pointer**) | **0.157649436** |
| net ΔS vs its own base | −6.3663e-4 | **−6.414241e-05** (bar −3.5e-6) |

Candidate `179,140 B`, sha `d2ad58ee28b84388a262bd5c8b11611a163dcc2694ad3c29a1283605a206b992`,
parse-back **PASS**, sealed at `/Volumes/APDataStore/pact/ddm_sa3/FIRE_ORDER_sa3.json`
(seal sha `2879720c0fb09e79…`, `SEAL_VALID`).

The honest shape of the result: **the mechanism is intact but the credit shrank 9.9×**, from
−6.37e-4 to −6.41e-5. sz1 had already banked most of what sa2's candidate was being paid for.
That is the delta-without-its-baseline law's second half — re-basing does not just move the
comparison, it *repossesses* the overlapping credit.

## 1. It is an IDENTITY, not a transfer — measured, then asserted in code

The charter said re-solve, do not transfer, citing qs4 (a Schur compensation carried onto a
different lattice cost +2.396e-4 pose, ~100× the seg win). Before re-solving I checked whether
the lattice actually differs. It does not:

| decoded object | rr4 | sz1 | same |
|---|---|---|---|
| carrier lattice (600×12 int12) | `ac6ab87fcd1c3944…` | `ac6ab87fcd1c3944…` | **yes**, max\|Δ\| = 0 |
| semantic body (36,040 B) | `b0d41ec904aca82f…` | `b0d41ec904aca82f…` | **yes** |
| canonical CPR1 carrier body (22,219 B) | `065fce08fc3d44e4…` | `065fce08fc3d44e4…` | **yes** |
| hpac body (17,952 B) | `e8c0cfd73d3275ad…` | `e8c0cfd73d3275ad…` | **yes** |
| semantic *stream* | 34,763 B | 34,243 B | no (−520) |
| token tail | 110,608 B | 109,897 B | no (−711) |

−520 + −711 = **−1,231 B**, which is exactly sz1's reported delta against generation 2. Both of
sz1's moves are **container** transforms over an unchanged decoded state.

The compensation is a function of (base lattice, base frame_1, edited frame_1). All three are
byte-identical here, so sa2's solution is valid **by identity**. This is not a licence to skip
the check: `assert_decoded_identity` recomputes all four digests at build time and **refuses**
unless they match, so the day sz1's successor actually changes the decoded state, the build
stops instead of silently repeating qs4. Each archive is decoded through *its own* shipped
runtime in a fresh interpreter — rr4's `residual_archive` refuses `reserved != 0` and cannot
read sz1 at all, and a shared interpreter would serve one root's modules to the other.

Structural precondition re-derived on this base, not inherited: `cpr1/inflate.py` is
**byte-identical** between rr4 and sz1 (`ff446edd9237148b…`), so `render_video` still writes
frame_1 from the semantic renderer and frame_0 from the carrier alone. d_seg invariance under a
frame-0 edit therefore holds here for the same reason it held for sa2 — SegNet reads
`x[:, -1, ...]`.

## 2. The token tail is edit-independent — measured, so sz1's fx2 credit is borrowable

sz1's −711 B lives in the token tail. That credit only carries if the tail is independent of
both the semantic edit and the carrier compensation. It is:

    tail sha  rr4 == S2 == sa2 == f74952f7190713d0…      sz1 == 59cc27c907d645c0…

The tail is byte-identical across the base, the S2-edited archive and sa2's compensated archive,
so it is a pure container leg and sz1's re-encoded tail drops in verbatim. `assert_tail_independence`
refuses the build if that ever stops being true.

## 3. The semantic byte-plane split is NEGATIVE on the edited body — the union is not the sum

sz1's second move byte-splits the fp16 metadata region for **−520 B** on the base body. The
obvious move is to keep it. Measured on real archive bytes, it is a **cost**:

| body | plain stream | split stream | Δ |
|---|---:|---:|---:|
| base semantic (36,040 B) | 34,763 | 34,243 | **−520** |
| **S2-edited semantic (38,514 B)** | **33,430** | **33,581** | **+151** |

The S2 edit changes the body *length* (36,040 → 38,514 B), so sz1's pinned region (offset 49,
length 8,284 — itself an argmax-tuned constant, +22 B over the format-derived (30, 8,284)) no
longer lands on the metadata. The permutation stays exactly invertible either way — that is why
`split_region`/`unsplit_region` are safe regardless — but its *profit* does not survive. More to
the point, quantizing the FiLM rows already removed the byte-plane redundancy the split was
harvesting; two credits over the same redundancy do not add.

The candidate therefore ships `reserved = 0`. Both variants were built and measured rather than
reasoned about:

| variant | archive B | Δ vs sz1 | net ΔS (conservative) | admits |
|---|---:|---:|---:|---|
| **candidate, split OFF** | **179,140** | **−790** | **−6.414241e-05** | **yes** |
| candidate, split ON | 179,291 | −639 | +3.640230e-05 | no |

The split would have flipped the candidate from admit to refuse. Had I inherited sz1's flag
because it was "already measured", this arm would have shipped a refusing archive.

## 4. What was rebuilt, and what it cost

    member = RX1M header(reserved=0)
           + hpac   13,515 B   (sz1, byte-identical to rr4)
           + semantic 33,430 B (S2-edited body, brotli, NOT split)
           + carrier 22,184 B  (compensated lattice, re-encoded CPR1 -> CAP1 -> packed)
           + tail   109,897 B  (sz1 fx2 tokens, verbatim)

| leg | bytes |
|---|---:|
| sz1 base | 179,930 |
| zero-compensation control (S2 edit, base lattice, overlay folded in) | 179,102 |
| **compensated candidate** | **179,140** |
| **compensation marginal cost** | **+38 B** |

sa2 measured +36 B for the same compensation on the rr4 container; +38 B here. The compensation
is still essentially free.

One runtime change is required and it is free: the shipped reader dispatches the carrier on a
pinned section length, so a re-solved lattice — whose Rice residual stream has a different
length — is refused. sa2's patch derives that length from the section's own u24 bit counts.
Pure framing arithmetic carrying no video-derived content, rule-118 clean, zero counted bytes
(inflate.py is unsized). The controls parse back under the **unpatched** sz1 receiver and the
candidates under the patched one, which is itself the cleanest demonstration that the patch is
doing exactly the one job claimed.

## 5. The admit arithmetic, against the live pointer

Base (sz1, [contest-CUDA T4 n600]): d_seg 2.9611e-4, d_pose 6.88e-6, 179,930 B.

| term | value | source |
|---|---:|---|
| rate (−790 B) | **−5.260286e-04** | measured archive bytes |
| seg (Δd_seg +1.72e-6) | **+1.720000e-04** | S2's measured advisory row; invariant under a frame-0 edit |
| pose (residual 4.893e-7, absolute transfer) | **+2.898862e-04** | sa2's n600 CPU solve |
| **net** | **−6.414241e-05** | bar −3.5e-6 → **admits by 18.3×** |
| projected S | **0.157649436** | |

**The pose-transfer model is the dominant uncertainty and it is unmeasured.** The base leg shows
**21.4× CPU-vs-T4 d_pose drift on identical bytes**. Two models bracket it:

| model | pose ΔS | net ΔS | projected S |
|---|---:|---:|---:|
| **absolute** (damage carries in absolute d_pose units) — **reported** | +2.8989e-4 | **−6.4142e-05** | 0.157649436 |
| relative (damage carries as a fraction of base d_pose) | +1.3750e-5 | −3.9304e-04 | 0.157320536 |

They differ by 21×. The absolute model is the conservative one *here* — `sqrt(10·x)` has a 4.63×
steeper marginal at sz1's T4 operating point than at the CPU instrument's — so it is the one this
arm reports. Note this is the opposite of the intuition that a lower base d_pose is safer: the
lower the operating point, the more expensive each absolute unit of pose damage becomes.

A CPU-fitted compensation transferring to T4 is **expectation, not measurement**. The mechanism
is physical — a real frame_0 image change nulling a real frame_1-induced pose shift in output
space — which is why I expect it to transfer, but only a T4 row settles it.

**Sanity control on the pricing function itself:** the uncompensated control (179,102 B, −828 B,
the *better* rate) prices at **net +8.330205e-02 → REFUSED**, reproducing sa1's original refusal
of this exact edit. A pricing function that could not reproduce the known refusal would not be
trustworthy on the admit.

## 6. The sa1 reopening — keep01's actuator authority is MEASURED, and it is not the constraint

sa2 named the ladder as the real prize and flagged the honest risk: keep01 is a **26×** pose
collapse where S2 was 6.7×, "a different and much harder ask… it should be measured, not
assumed." Priced against the live pointer, the ask is sharper than the 26× headline suggests:

| row | Δbytes vs sz1 | rate credit | damage (CPU d_pose) | vs S2's damage | **required cancellation** |
|---|---:|---:|---:|---:|---:|
| S2 | −813 | −5.4134e-4 | 8.3908e-4 | 1.00× | 99.9261% |
| **sm3r_keep01** | **−2,369** | **−1.5774e-3** | **3.7265e-3** | **4.44×** | **99.9479%** |

keep01 needs **more** cancellation than has ever been achieved (sa2's 99.9417%) against **4.44×
more damage**. Both moves are adverse. Measured, n = 16 **seeded-random** pairs (never a prefix —
prefix bias on the pose axis measures 2.5–4.2× *harder* than the population, exactly the
false-negative shape a NO-GO here would take), identical solver and receiver realization:

| quantity | value |
|---|---:|
| mean damage | 1.857692e-03 |
| **mean residual** | **−3.542448e-10** |
| **achieved cancellation** | **100.000019%** |
| required | 99.9479% |
| allowed residual | 1.940020e-06 |
| **residual vs ceiling** | **5,476× under** |
| pairs ending better than base | 9 of 16 |

Projected net **−1.100848e-03 → S 0.156613** — **17.2× better than the S2 candidate.**

**The representativeness caveat, stated plainly:** the subset's mean damage is **0.4985×** the
n600 damage, so the subset is a *weak* estimator of the population mean and the projection above
must not be read as a measurement. What does not depend on the subset mean is the per-pair
evidence: pair 408 cancelled **100.007%** of a **1.02e-2** single-pair damage and pair 28
**100.014%** of **7.81e-3** — respectively **2.7×** and **2.1×** keep01's entire n600 mean damage.
The actuator is demonstrably not saturated at keep01's damage scale. **Actuator authority is not
the binding constraint.** The full n600 solve is running (resumable, per-pair checkpointed) and
is owed before keep01 is a candidate.

### The ladder, ranked by credit on the sz1 container

`Δbytes ≈ (sa1 archive bytes) − 711` (fx2 tail carries; split measured negative). Rows above S2
all carry larger credit than the sealed candidate and are now worth compensating:

| rank | row | Δbytes vs sz1 | rate credit | status |
|---|---|---:|---:|---|
| 1 | `sm3r_keep01` | −2,369 | −1.5774e-3 | **authority MEASURED feasible; n600 solve running** |
| 2 | `sm3r_keep06` | −2,071 | −1.3790e-3 | unowned |
| 3 | `sm3r_keep09` | −1,959 | −1.3044e-3 | unowned |
| 4 | `sm3r_keep15` | −1,742 | −1.1599e-3 | unowned |
| 5 | `sm3r_keep20` | −1,588 | −1.0574e-3 | unowned |
| 6 | `sm3r_keep37` | −1,004 | −6.6852e-4 | unowned |
| 7 | **`S2_film23_q2_top3_q3`** | **−790** (built) | −5.2603e-4 | **SEALED, this arm** |

Ranks 2–6 have no measured advisory pose row, so their required cancellation cannot be derived
yet; only keep01 and S2 have both legs. Their seg cost is also unmeasured — keep01's +4.77e-6 is
2.8× S2's +1.72e-6, so seg is not free on the SM3R family and grows as the prune deepens.

## 7. A custody incident I caused, and the structural cure

**I contaminated the sz1 submission packet.** At **2026-08-18T13:16:25Z** my first identity probe
(`.omx/tmp/sa3/probe_identity.py`) imported the packet's receiver modules **in place**, with
neither `PYTHONDONTWRITEBYTECODE` nor `-B`. That wrote **15 CPython-3.13 `.pyc`** into the hashed
runtime tree, plus 18 AppleDouble twins = the 33 files in MAIN's manifest:

    cpr1/__pycache__/carrier_codec.cpython-313.pyc
    runtime/__pycache__/{__init__,baseline,bits,carrier_repack,compensation_overlay,
                         frame0_selector,ihs2,residual_archive}.cpython-313.pyc
    runtime/entropy/__pycache__/{__init__,adaptive_ans,coefficient_ar1_codec,
                                 coefficient_predictor,rc64,renderer_weight_codec}.cpython-313.pyc

My own tree listing records `carrier_codec.cpython-313.pyc` at sha `8492a48f8118…`, matching
MAIN's manifest entry exactly. Two real consequences, both MAIN's diagnosis and both correct: the
pinned runtime-tree sha `0d0fc008d6a37bd5…` cited by report.txt / README / receipt r5 stopped
verifying, and the `.pyc` embed absolute local paths — a public-hygiene violation inside a
submission artifact. MAIN removed them at 13:27:21Z and proved restoration.

The cure is structural, not procedural. `PYTHONDONTWRITEBYTECODE` alone only prevents the
*symptom*; a copy removes the whole class, because the packet is then never on any interpreter's
import path at all. Every runtime import in this arm now goes through `scratch_runtime_copy`, and
`assert_packet_pristine` **fails the build closed** if the packet carries any residue before or
after. Belt-and-suspenders: `-B` plus `PYTHONDONTWRITEBYTECODE=1` on every subprocess.

**Denominator, as required:** `/usr/bin/find` over the packet directory now reports
**0 `*.pyc`, 0 `__pycache__`, 0 `._*`, out of 39 total files** — verified before the rebuild,
after the rebuild, and again at hand-off. The rebuild reproduced the identical archive sha
`d2ad58ee28b84388…` with the packet never imported, which is both a determinism repeat and proof
the cure changed nothing about the result.

## 8. Honest limits

1. **No score.** Every d_pose here is sa2's frozen CPU-torch PoseNet; every byte is exact. Only
   `upstream/evaluate.py` on contest hardware is an authority. The pointer is unmoved.
2. **The T4 transfer is unmeasured** (§5). This is the single largest risk and the reported
   number is the conservative side of a 21× bracket.
3. **d_seg is asserted structurally, not re-measured.** The argument is exact — SegNet reads
   frame_1, the compensation touches only frame_0, and `cpr1/inflate.py` is byte-identical to the
   tree sa2 verified — but the candidate's d_seg has not been run. MAIN's row measures it.
4. **The pose legs are sa2's, valid here by the §1 identity, not re-measured.** I re-derived the
   *precondition* and assert the identity in code; I did not re-run sa2's 600-pair solve for S2.
   If §1's gate ever fails, every pose number in §5 is void by construction.
5. **keep01's projection rests on n=16 with a 0.4985 representativeness ratio** (§6). The
   feasibility read leans on the per-pair existence proofs, which do not depend on that ratio;
   the *projected score* does, and should not be quoted as measured.
6. **Full-frame inflate was not run.** Parse-back is verified at the carrier/semantic section
   level through the shipping receiver in a fresh interpreter; the token tail is byte-identical
   to sz1's, so the ~19-minute decode path is unchanged by construction.
7. **The candidate is not the frontier.** −6.41e-5 is **0.83%** of the 7.7136e-3 gap from
   0.157714 to sub-0.15. keep01, if it lands, is 14.3% of that gap. Neither reaches the goal.

## STORES CONSULTED

`ddm_sa2_compensated_semantic_edit_20260818.md` (the mechanism, the solve, the +36 B route) ·
`ddm_sa1_advisory_adjudication_20260818.md` (the base leg, keep01's and S2's measured rows, the
21.4×/1.44× drift) · `ddm_sa1_semantic_carrier_representation_attack_20260817.md` (the 16-row
ladder) · `ddm_sz1_semantic_metadata_split.py` (the split's derived-vs-tuned profiles and its own
"±20 B is Brotli alignment noise" caution) · `ddm_qs5`/`ddm_qs4` (re-solve-in-compile; the
cross-regime transfer disaster) · `ddm_fx1/fx2` (the token mixer, the tail) · live source:
`gen3_sz1_composed_split/{archive_manifest.json, report.txt, runtime/residual_archive.py,
cpr1/inflate.py}`, `upstream/modules.py`. Memory: `a_delta_without_its_baseline_is_unanchored…`,
`never-price-a-union-as-the-sum-of-its-legs`, `m96` (prefix bias inverts by axis), `m87`.

## RETAINED

* build: `/Volumes/APDataStore/pact/ddm_sa3/build/` — all four archives, `SA3_REBASE.json`
  (identity gate, tail gate, pristine gates, per-variant admit arithmetic), expected-code arrays
* generation: `/Volumes/APDataStore/pact/ddm_sa3/generations/sa3_rebased_sz1/` — archive +
  patched runtime + re-pinned `inflate.py`
* seal: `/Volumes/APDataStore/pact/ddm_sa3/FIRE_ORDER_sa3.json` (sha `2879720c0fb09e79…`)
* keep01 authority: `/Volumes/APDataStore/pact/ddm_sa3/keep01_authority/sm3r_keep01/` — per-pair
  `RESULT.json` (base/final codes, every evaluated code row and its PoseNet-6 vector),
  `AUTHORITY.json`

## NEXT_IF_RESUMED

* `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN. Fire the advisory n600 leg on
  `FIRE_ORDER_sa3.json` against the sz1 base leg, recompute the decomposition on the same
  instrument, admit only on measured net ΔS < −3.5e-6. Four falsifiers are pre-registered in the
  seal.
* **Running, owner: MAIN to harvest** — the keep01 n600 solve
  (`experiments/ddm_sa3_keep01_actuator_authority.py --pairs 600`, ~3 h, per-pair checkpointed
  and resumable). On completion, byte-close it exactly as §4 did for S2 — the builder
  parameterizes cleanly on `semantic_source` — and seal. Expected net ≈ −1.10e-3, **17× this
  arm's candidate**. This is the higher-value row; the sealed S2 candidate is the safe one.
* Unowned: ranks 2–6 of §6 have no advisory pose row, so their required cancellation is
  underivable. One advisory n600 leg per row unlocks the whole ladder; keep01 and S2 already
  bracket the slope.
* Unowned: the T4 transfer question (§5). A single T4 row on the sealed candidate settles a 21×
  bracket that currently gates every compensated row on the ladder.
