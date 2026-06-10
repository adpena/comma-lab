# Why our substrate work was broken derivatives — and the redirect (operator, 2026-06-10)

**Operator (verbatim, 2026-06-10):** *"I asked numerous times over the past few months to reverse
engineer and deconstruct all details of PR95 and other leading PRs but I don't understand how our
reconstruction is still so broken and how all of our substrate work has basically been broken
derivatives, when I emphasized with the strongest emphasis that nibbling on derivatives is antithetical
to our mission — original bold frontier score-lowering research, not nibbling — and that those methods
are important for calibrating and measuring drift and for creating tools. I'm confused why our stuff is
so broken when PR95's full codebase, submission, thinking, writeup, blog, everything are available."* +
*"if a technique would be extreme-optimal but doesn't play well with our current lane/substrate/approach,
take time to research, detour, design, experiment and test the frontier-score-lowering possibility with
a full stack designed around it for max synergy."*

## The answer (the root process failure, evidence-cited)
We had PR95's full proven code and **re-implemented it into our OWN shared MLX harness instead of
vendoring the proven code verbatim.** The 2026-06-10 audits pinpoint a SINGLE shared point of failure
that infected the entire fleet:
- **#68** — `tac.substrates._shared.mlx_score_aware.bundle.py` defaults every SegNet/PoseNet objective
  weight to **0.0** (M-loss → "score-aware" runs were silently recon-MSE-only / scorer-blind) AND the
  default decoder is **skip-free** PixelShuffle+sin, missing PR95's bilinear-skip+HF-refine (M-arch →
  mean-field blur → argmax collapse).
- **#77** — our reproduction ran **AdamW** through stages 1–7 (Muon only in stage 8), so the
  ill-conditioned score-aware loss never descended (grad_norm 6.8e6 → clip-to-1.0 → ~1.5e-7 step).
- **#75** — proof the eval was fine (reproduced PR95's 0.19871 bit-exact); the d_seg≈0.50 plateau was
  the broken loop, not a wall.

**Because all 30+ substrates were built ON that one shared harness, one config/arch/optimizer bug broke
all of them at once.** They were derivatives-of-a-derivative on a broken base — the "vehicle names
outran vehicle implementations" failure the Vehicle OS already named, realized at the shared-harness
layer. The "reuse a canonical helper" reflex became a single point of failure that silently corrupted
every score-aware run. We never ran PR95's actual proven loop; we ran a buggy lookalike.

## The mission correction (binding)
Reproduction is a **TOOL**, not the mission:
- PR95 reproduction = the calibrated reference + drift measurement + a working-loop tool. Vendor it
  VERBATIM (don't re-implement) so it is *trusted*. That is what reproductions are FOR.
- The MISSION = ORIGINAL bold frontier score-lowering — our OWN small basis (the capstone), a
  class-shift, our own summit. NOT nibbling on broken reproductions, NOT post-hoc-compressing the
  frozen frontier (the 7 no-moves were calibration, not the mission).
Spending months stuck fixing a broken derivative IS the antithesis of the mission. The redirect:

## The redirect (two clean moves + the synergy-detour mode)
1. **Vendor the proven code as the tool** (#76): run PR95's `hnerv_muon/src/` loop VERBATIM as the
   calibrated reference. Stop re-implementing proven code into the shared harness.
2. **Build the original capstone CLEAN** (#78): our own small VQ-NeRV-class basis as a DEDICATED full
   stack — NOT on the broken shared harness — per UNIQUE-AND-COMPLETE-PER-METHOD. Each method gets the
   optimal engineering for ITS math, bound into one coherent package; no force-fit into a shared base
   whose defaults can silently break it.
3. **The synergy-detour mode (standing directive):** if a technique is extreme-optimal but doesn't
   synergize with the current lane, do NOT force-fit it — DETOUR: research + design + experiment + test
   the frontier-score-lowering possibility with a FULL STACK designed around it for max synergy. The
   shared-harness false economy is exactly what this prevents.

## The operator's actual intent (clarified 2026-06-10): a 1:1 MLX PORT, not an inspired harness
*"I didn't want a PR95-inspired harness I wanted a 1:1 MLX port ... tested and iterated and optimized
for determinism and correctness and optimization."* This is the precise gap. We built a loose
"PR95-inspired" MLX harness with its own (wrong) defaults. The operator wanted a **faithful 1:1 MLX
port of PR95's `hnerv_muon`** — and crucially, one with a **torch-parity GATE**: the MLX port must
produce bit-/score-identical output to PR95's torch loop on the same input. That gate is the fidelity
discipline that was missing — it would have caught EVERY divergence the moment it appeared:
- scorer-weights=0.0 → fails parity (PR95's weights are nonzero) → caught.
- skip-free decoder → fails parity (PR95 has bilinear-skip+HF-refine) → caught.
- AdamW-not-Muon-throughout → fails parity (PR95 runs Muon) → caught.
A 1:1 port is defined by its parity test, NOT by "it looks like PR95." MLX-first was correct; the
missing piece was **1:1 fidelity + a parity gate + tested determinism + correctness, THEN optimization.**

## Durable lesson (so this never recurs)
- **Vendor proven code verbatim; never re-implement it into a shared harness with its own defaults.** A
  shared harness is a single point of failure — a wrong default (scorer-weights=0.0) silently breaks
  every consumer. The Vehicle OS rule (no vehicle until its `vehicle_fidelity_manifest.verify()` passes)
  must include a **fidelity parity check against the vendored proven loop** before any substrate trusts
  the shared harness.
- **Original ≠ derivative-of-a-derivative.** The mission is original frontier work on a clean stack;
  reproductions are tools that must be FAITHFUL (vendored) to be useful as drift references.
- **A "no-move" on the frozen frontier is calibration, not the mission.** The 7 no-moves correctly
  closed the post-hoc levers, but the mission is the original retrained basis — pivot firepower there.
