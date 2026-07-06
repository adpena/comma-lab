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

## Scenes  (intended order: 1 = abstract intro → 2 = grounded on real data)

| # | file | class | shows | status |
|---|------|-------|-------|--------|
| 1 | `scene01_separatrix.py` | `Separatrix` | **INTRO** — softmax τ→0 → Laguerre/power-diagram argmax partition → the codim-1 **separatrix** that IS d_seg · `τ = ε = ℏ` | ✅ first cut |
| 2 | `scene02_hardest_frame.py` | `HardestFrame` | fast-forward the REAL contest video → slam to the hardest frame (196) → real SegNet argmax → separatrix → margin field (= Fisher metric, Pearson 0.978) | ✅ first cut |

Scene 2 depends on assets from `scenes/_prep_hardest_frame.py` (run once; reads the
`gt_n600.npz` cache, writes `assets/` — gitignored, rebuildable).

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
