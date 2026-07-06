# Level-set witness — 3Blue1Brown-style animations

Standalone, self-contained [Manim](https://www.manim.community/) animations of the
**beautiful, MEASURED math** behind our level-set task-space witness. Every scene
shows real math from our own findings (deepmath `#284`, the unified level-set
flow, the triality) — faithful, not decorative.

## Setup (one-time, isolated — does NOT touch the main `.venv`)

```bash
cd experiments/manim_levelset
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python manim
# (optional, for LaTeX equation typography): brew install --cask basictex
```

## Render

```bash
cd experiments/manim_levelset
.venv/bin/manim -qm scenes/scene01_separatrix.py Separatrix     # 720p, fast
.venv/bin/manim -qh scenes/scene01_separatrix.py Separatrix     # 1080p, final
```
Output lands in `media/videos/<scene>/<res>/<Class>.mp4` (gitignored — rebuildable).

## Scenes  (series order: 00 idea → 01 separatrix → 02 real frame → 03 screw)

All scenes share `_style.py`: the design system (Space Grotesk / Inter / JetBrains
Mono, deep-black canvas, comma-coral accent, real LaTeX math) + a **layout grammar**
(reserved top/stage/bottom bands so nothing overlaps) + a **pacing rhythm**
(`T_*`/`HOLD*` constants — one idea per beat).

| # | file | class | shows |
|---|------|-------|-------|
| 00 | `scene00_witness.py` | `Witness` | **cold open** — the contest scores what two frozen nets SEE; builds `S = 100·d_seg + √(10·d_pose) + 25·bytes/N`; "code the task, not the picture"; hands off to the series |
| 01 | `scene01_separatrix.py` | `Separatrix` | softmax τ→0 → Laguerre argmax partition → the codim-1 **separatrix** that IS d_seg → **resolves into the real openpilot/comma10k segmentation** · `τ=ε=ℏ` |
| 02 | `scene02_hardest_frame.py` | `HardestFrame` | fast-forward the REAL video → hardest frame (196) → openpilot segmentation → separatrix → margin = Fisher metric (Pearson 0.978) |
| 03 | `scene03_screw.py` | `Screw` | 3D **se(3) screw** — Chasles; the ego-motion is one twist ξ; the SAME ξ is d_pose AND warps the partition (d_seg) — encode it once |

Scenes 00–02 depend on assets from `scenes/_prep_hardest_frame.py` (run once; reads
the `gt_n600.npz` cache, writes `assets/` — gitignored, rebuildable, real comma10k colors).

### Roadmap (candidate next scenes — all faithful to measured findings)
- **The level-set φ** — the signed-distance function whose zero-set is the boundary; the eikonal `‖∇φ‖=1` flow (Osher–Sethian), our viscosity-solution frame.
- **The margin IS the Fisher metric** — flat interior (argmax stable → dark) + anisotropic bright boundary annulus; Fisher-curvature ↔ (−margin), Pearson **0.978**.
- **One twist warps the world** — the se(3) ego **screw** ξ: the SAME twist that moves the partition (d_seg) IS the pose (d_pose). Chasles + dual-use.
- **Lane erasure** — finest-scale features (lane dashes) dropping below the argmax margin (spectral bias / persistence); Gibbs ringing as the spatial dual.
- **The triality** — DAG ↔ DSL ↔ equations as three cyclically-related shadows of one campaign (Spin(8)).

## Design rules
- **Correct math only** (NO-FAKE applies here too): every visual must be a real,
  reproducible computation of our findings — no pretty-but-wrong.
- Isolated venv + gitignored artifacts (rebuildable; disk-hygiene non-negotiable).
- Pango/Unicode typography works today; LaTeX (`MathTex`) is a fast-follow once
  BasicTeX is installed, for full equation beauty.
