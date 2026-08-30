# ddm_ni1r — NR1 K32's distortion is MEASURED at last: byte-feasible and catastrophic, and my own prediction is refuted 3× harder than its falsifier anticipated

**Date:** 2026-08-30
**Status:** `MEASURED_DEAD_ON_DISTORTION` — the ni1 (2026-08-22) queued measurement, fired and harvested $0.
**Axis:** `[macOS-CPU frozen-scorer advisory]`, `score_claim=false`, `promotable=false`,
`score_axis=cpu_env_mismatch_advisory`. Only `upstream/evaluate.py` on the exact bytes at a
contest axis is authority; this is an advisory row and is used ONLY as a distortion-regime read.
**Run:** counter 701, pid 13831, rc=0, 820 s (inflate 395.7 s + evaluate 416.2 s), n=600.
**Attempt dir:** `/Volumes/APDataStore/pact/ddm_ni1_nr1_k32_receiver_distortion/advisory_n600_20260830T051434Z/`
**Archive:** `.../build_r4/runtime/archive.zip`, **122,250 B**,
sha256 `fe7fe8058376543d5832912e691214969680fea5d85e125e861e9700c5ca534e`
(verified byte-exact against ni1's own recorded value at fire time; pin preflight `f3d6aba3e1` clean).

## 1. Why this ran at all — a correction, not a plan

`ddm_ni1_nr1_k32_receiver_distortion_20260822.md` byte-closed NR1 K32 and left its n600
`d_seg` / `d_pose` / `S` **NOT MEASURED**, verdict `INCONCLUSIVE`, MAIN-fire-only. It sat eight days.

It surfaced on 2026-08-30 from the `#1187` correction: that row's headline
("NR1 IS DEAD ON DISTORTION — MEASURED at 349× its own admission bar") cited a number that is a
**proxy-understatement factor** (parameter/token agreement vs evaluator), corrected at source five
days earlier in `ddm_df1_dddb_field_20260824.md:547-548`. The correction made NR1's distortion
UNMEASURED again — which made it the one cheap decisive measurement on the board.

## 2. THE RESULT — recomputed FROM COMPONENTS (#877), never the rounded display

The report's `final_score` field displays **27.8**; the canonical S from components is
**27.797427174805083** (`score_rounding_abs_delta` 2.5728e-3 — the display is wrong by more than
the entire sub-0.12 gap, exactly why #877 binds).

| term | value | vs lb1 pointer |
|---|---:|---:|
| archive | **122,250 B** | **15,736 B UNDER** the 137,986 B sub-0.12 cap ✅ |
| rate = 25·B/37,545,489 | 0.081401257 | — |
| d_seg 0.07584291 → seg | **7.584291** | **376.6×** lb1's 0.00020139 |
| d_pose 40.52867508 → pose | **20.131735** | **6.36e+06×** lb1's 6.37e-6 |
| **S** | **27.797427174805083** | — |
| **distortion (seg+pose)** | **27.716026** | **985.6×** lb1's 0.028120 |

## 3. THE PREDICTION IS REFUTED — and by more than its own falsifier required

`ddm_xo1_cross_successor_object_20260830.md` §OPTIMAL FORM pre-registered:

> the sy2 object-change law + THE CROSS jointly predict that a successor exists in the
> intersection {byte-feasible by construction} ∩ {distortion regime inherited from dx2}, and that
> **NR1 K32 is the one measured body sitting in it**.
> FALSIFIER: NR1's distortion measures in the born object's regime (≳0.3, i.e. an order above
> lb1's 0.028120) rather than dx2's ⇒ the intersection is measured EMPTY at n=2.

**The falsifier FIRED, and overshot.** NR1's distortion is not merely "in the born regime" — it is
**84.57× BEYOND the born object's 0.327712.** The prediction was wrong in the direction it was
written to be able to be wrong in. Counted plainly, as the charter demanded.

## 4. What the three measured byte-feasible bodies now say TOGETHER

| body | archive B | under cap? | distortion (seg+pose) | axis |
|---|---:|---|---:|---|
| born-small / qbt2b r10 | 121,928 | +16,058 | 0.327712 | `[macOS-MPS n32-HT advisory]` |
| **NR1 K32** | **122,250** | **+15,736** | **27.716026** | `[macOS-CPU frozen-scorer advisory]` |
| lb1 pointer | 180,083 | −42,097 | **0.028120** | `[contest-CUDA T4 n600]` authority |

**THE INTERSECTION IS MEASURED EMPTY.** No measured body is both byte-feasible and in the dx2
distortion regime. THE CROSS's two halves remain held by two different objects.

**And the sharper read, which the charter did not anticipate:** born-small and NR1 sit **322 B
apart in rate** — 0.26% — and **84.57× apart in distortion.** So "small" is NOT one regime with a
rate-distortion ordering inside it. Byte feasibility does not predict distortion AT ALL among the
small bodies; they are a scattered set of different constructions that happen to share a size.
Any successor argument of the form "build it small and the distortion will follow the small-body
trend" has no trend to follow — n=2 spanning two orders of magnitude is not a trend.

**Mechanism, named:** NR1's d_pose 40.53 means it carries **essentially no pose at all** (lb1:
6.37e-6). Its pose term alone is 20.13 S — 718× the entire sub-0.12 gap. The seg term (7.58)
is separately catastrophic. This is not a body needing a finisher; it is a body whose scored
quantities were never carried.

## 5. Disposition + the #1187 reconciliation (APPEND-ONLY, Catalog #110/#113)

- **NR1 K32: `MEASURED_DEAD_ON_DISTORTION`.** `verdict_scope: INSTANCE` — the NR1 K32 build_r4
  archive at 122,250 B, advisory axis. This does NOT close the NR1 construction family; it closes
  this member, with a receipt instead of a mis-read proxy.
- **#1187's conclusion is now independently established, but its evidence never supported it.**
  Both halves of my 2026-08-30 correction stand: the 349× WAS mis-cited (it is a
  proxy-understatement factor, not a damage÷credit ratio) and NR1's distortion WAS unmeasured at
  the time the row asserted otherwise. A conclusion that turns out true does not retroactively
  make its wrong evidence right — the row was a false-measurement claim that happened to point the
  correct way. Consumers `ddm_af1:131` and `ddm_bs4y:49` still carry the stale 349×; neither may
  cite it as a distortion ratio.
- **`ddm_xo1` (LIVE) consumes this** per its own charter clause: the attempt dir is named in its
  HARD CONSTRAINTS with instructions to recompute from components and label advisory. Its
  prediction is refuted; its §3 intersection question now has a measured-empty answer at n=3.
- **Env label, honestly:** `env_mismatch.reason = uv_group_not_declared` — the evaluation ran on
  the repo venv (torch 2.12.1 / py3.13.12) rather than the pinned
  `upstream_eval_mirror_20260815/uv.lock` reference. That is a DECLARATION gap; it correctly holds
  the row at advisory (where it already was by construction) and is not a numerical-corruption
  claim. No plausible environment artifact spans 985×. The regime read is safe; a score claim
  would not be, and none is made.

**Own-vehicle frontier: lb1 — S 0.14803010583079396 @ 180,083 B [contest-CUDA T4, n600], UNMOVED.**
Sub-0.12 gap 0.028030; demand 42,097 B at current distortion 0.028120.
