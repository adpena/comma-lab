# The #1092 drain is on a different vehicle than the frontier — and lever 2's rate payoff is negative at every fidelity

**Status:** MEASURED (2026-08-17, MAIN, $0, read-only on retained checkpoints).
**No score claim. Frontier untouched.** Two findings, both routing-decisive, both cheap.

## FINDING 1 — two architectures, not two checkpoints

Read at source from each checkpoint's own `run_identity` / `state_dict`:

| | frontier `hv1 ep0634` | `ce1` / `EF3000` / `FRD077` |
|---|---|---|
| trainer | `tools/train_ddm_cl1_hpac_capacity_mps.py` | `src/tac/pr130_lift/train_semantic_quantized_resumable.py` |
| identity schema | `ddm_cl1_hpac_capacity_run_identity.v1` | (semantic renderer) |
| tensors / params | 37 / **39,375** | 38 / **66,339** |
| shape | `conv_a · conv_b1 · conv_b2 · conv_past · spm_dw · spm_pw · head` (+ per-tensor `exponent`/`bit_depth`) | `blocks.{0..3}.{dw,film,norm,pw}` + `coord_mix` + `frame_embed` + `token_embed` + `head` |
| FiLM tensors | **NONE** | 8 (`blocks.{0..3}.film.{weight,bias}`) |
| profile | `rx2_mc36`, channels 64, spm on | `width=96, blocks=4, frame_dim=8` |

**Every lever in the #1092 drain is an argparse flag on the SECOND trainer.** The frontier
vehicle has no FiLM rows at all, so `--film-row-dropout` / `--film-row-dropout-protect-top`
cannot be applied to it even in principle.

This does not invalidate the FRD077 result — seg-neutral is seg-neutral on the vehicle it ran
on — but it sharpens the scope: my ef6000 memo said "NOT the hv1 frontier vehicle," which reads
as *a different checkpoint*. It is a **different architecture**. The transfer question I filed as
"unmeasured" is not a measurement away; for the FiLM levers it is structurally undefined.

The semantic line is a live, named route — the hot state's own rate verdict says the remaining
routes run through "joint descent: carrier retrained with pose in the loop, **or the semantic/FiLM
field**." So the drain is work on a *replacement-vehicle candidate*, not a modification of the
frontier archive. It has to be labelled that way in every row it produces.

Genus: task #917 ("the lever instruments all point at a retired vehicle"). Same failure, one layer
down — not the instruments this time, the levers themselves.

## FINDING 2 — lever 2's rate payoff channel is negative, before any training

`--carrier-rank-penalty` (F4) adds `w · Σ_tensors ‖σ‖₁/‖σ‖₂` — a **stable-rank** (spectral
concentration) penalty on `--carrier-tensors`. Its payoff story is: concentrate the spectrum →
factorize low-rank → ship fewer bytes. Priced the factorization directly on the EF3000 EMA shadow:

Rank-r factorization of an (m,n) matrix costs `r(m+n)` vs `mn` dense ⇒ pays only for `r < mn/(m+n)`.

| tensor | shape | dense | break-even r | r@99% energy | cost @99% |
|---|---|---:|---:|---:|---:|
| `blocks.0.pw.weight` | 96×96 | 9,216 | 48 | **73** | **152% of dense** |
| `blocks.1.pw.weight` | 96×96 | 9,216 | 48 | 74 | 154% |
| `blocks.2.pw.weight` | 96×96 | 9,216 | 48 | 72 | 150% |
| `blocks.3.pw.weight` | 96×96 | 9,216 | 48 | 72 | 150% |
| `coord_mix.weight` | 96×100 | 9,600 | 49 | 74 | 151% |

At 99.9% energy: r ≈ 85–87 (177–179%). At 99.99%: r ≈ 90–92 (188–192%).
**Factorization loses at every fidelity level, on every candidate tensor.**

**The trap I nearly walked into:** these same tensors report `stable_rank ≈ 8 of 96` — 8% of full
rank, which reads as "already strongly low-rank," and is exactly the functional F4 minimises.
But `‖σ‖₁/‖σ‖₂` is set by the few LARGEST singular values, while truncation cost is set by the
TAIL. A spectrum can be concentrated (stable rank 8) and still need 73 components to reconstruct.
Had I quoted stable rank as the payoff signal, the lever would have looked ready to fire.

**Scope, honestly:** this prices the FACTORIZATION channel and refutes it (INSTANCE: EF3000 EMA
shadow, these 5 tensors). It does NOT refute the lever. A second payoff channel exists and is
unmeasured: a concentrated spectrum may cost fewer CODED bytes through the deployed quantiser +
entropy coder without any factorization. That channel needs its own measurement — the coded size
of the archive under a rank-penalised vs control checkpoint — and it is not the one the lever's
own name implies.

**Recall check (not re-derived):** `ra2` closed carrier rank/refit at FAMILY scope 08-16 on the
**frontier** archive — six treatments, closed on DISTORTION not rate (rank-4 returned 14,709 B =
102.1% of the rate bar, then the score-relevant functional missed by 1,498×–3,139× over the whole
sphere). That is POST-HOC refit on hv1. Training-time penalty on the semantic renderer is a
different vehicle AND a different mechanism, so ra2 does not close it — but it does mean the
rank family has now failed on both axes it was supposed to win on, in two different places.

## Routing

**Lever 2 does NOT fire.** Its named payoff is negative by arithmetic; firing a ~25-min Metal A/B
to learn a lever is free would repeat FRD077 with the answer already known.

**Next in the drain: `--weight-qat-q3q4` (F2).** It is the one remaining lever with a payoff that
is already MEASURED and BANKED: mz2 retained a mixed q3/q4 candidate at **−823 B**, explicitly
unscored, with distortion as its named blocker — and F2 is exactly the cure for that blocker
(train THROUGH the deployed mixed grid, the eval-roundtrip principle applied to weight
quantization). Byte win measured, distortion the open question, lever aimed at the open question.
That is the shape a fire should have.

Remaining after F2: `--weight-perturb-robustness` (+`--weight-perturb-shape`,
`--film-critical-multiplier`) · `--distill-weight`/`--distill-max-seg` · `--film-row-dropout-protect-top`
(inherits lever 1's blocked payoff path) · `--fixed-zero-mask` (a sparsity-PRESERVATION lever, not
an ablation actuator — its one consumer pins already-zero init weights).

**Owed on every row this drain produces:** the vehicle label. A verdict on the semantic renderer is
not a verdict on the frontier, and the two are further apart than "different checkpoint."

## FINDING 3 — CORRECTION, same turn: "never-fired" was inferred from argparse, not from the runs

⚠ I routed `--weight-qat-q3q4` as "the next fire, one of the 12 never-fired levers." **It is ON in
every `ce1` arm.** Read from the launch manifests, not the flag list:

| arm | `--weight-qat-q3q4` | other levers passed |
|---|---|---|
| `CE0` | **ON** | `--band-objective-weight 0.0` (inert) |
| `EF0` | **ON** | `--band-objective-weight 0.0` |
| `EF3000` | **ON** | `--band-objective-weight 0.0` |
| `EF6000` | **ON** | `--band-objective-weight 0.0` |
| `FRD077` | **ON** | `--band-objective-weight 0.0`, `--film-row-dropout 0.077` |

My inference was: argparse default is `False` ⇒ the lever has never fired. **Default-off in the
parser is not evidence about the runs; the launch manifests are.** F2's real status is not
NEVER-FIRED, it is **ON-WITHOUT-A-CONTROL** — which is worse, because its contribution is baked
silently into every `ce1` number I have quoted today: the allocation ladder (control +8,654 /
CE0 +4,852 / EF0 +636), the EF6000 depth curve, and the FRD077 verdict. None of those isolate it.

**What SURVIVES this correction:** the FRD077 A/B is genuinely single-variable. Verified by argv
diff — the *only* token difference between `EF3000` and `FRD077` is `--film-row-dropout 0.077`.
`q3q4` is present in both, so it is held constant and cannot confound that comparison. The
seg-neutral verdict stands; so does the ladder's ordering (q3q4 constant across all three arms).

**What CHANGES:** F2 is not a fire, it is an **owed control** — the OFF arm has never been run, so
"train through the mixed q3/q4 grid" has no measured contribution on this vehicle at all. And
mz2's banked −823 B q3/q4 candidate cannot be attributed to it without that control.

Genus: this is today's third instance of *asserting a property of the live system from a static
surface instead of the runtime record* — after row-norms-instead-of-ablation and
stable-rank-instead-of-truncation. The cure is the same each time: read what actually ran.
