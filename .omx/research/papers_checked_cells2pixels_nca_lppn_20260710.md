# papers-checked — Cells2Pixels: NCA + implicit decoder (operator link 2026-07-10)

`research_only=true`

**Source:** https://cells2pixels.github.io (#growing) — "Neural Cellular Automata: From Cells to
Pixels", SIGGRAPH 2026, Pajouheshgar/Xu/Abbasi (EPFL) + Mordvintsev (Google) + Jakob/Süsstrunk.
arXiv 2506.22899. Code github.com/TheDevilWillBeBee/Cells2Pixels.

STORES CONSULTED: #146 NCA shared-rule generalization gate (AMBER lineage) · #143 NCA feasibility ·
#308 grids-for-bulk + INR-for-annulus hybrid · #395 texture trunk (W=(G,ξ,T)) · #211
amortized pre-seeding · papers_checked ledger.

**Their method:** NCA evolves cell states on a COARSE grid (learned local update rule =
self-organization); a lightweight shared implicit decoder (LPPN: (cell state, local coord) → color)
renders at ARBITRARY resolution. 2D/3D/mesh domains, real-time.

**Mapping to our stack (the architecture is structurally OURS):**
1. **Coarse-state + shared implicit decoder = the witness family**: their (cell state, local
   coord)→LPPN is our (mod vector, coord)→FiLM-trunk — and HNeRV's (latent, coord)→decoder. Their
   contribution = the STATE EVOLVES by a learned local rule instead of being stored per-cell.
2. **The NCA rule as a GENERATOR is our #146 AMBER question**: ONE small shared rule + cheap seeds
   amortizes per-frame content — #146 MEASURED the make-or-break (33K rule holding d_seg across
   16-48 frames). Cells2Pixels adds the missing piece our #146 arm lacked: the coarse-grid NCA +
   arbitrary-res decoder SPLIT, which cuts the NCA's compute/state cost while the LPPN carries
   resolution. If AMBER is ever reactivated (P10 queue), THIS is the reformulation to try —
   coarse NCA grid + our existing trunk as the LPPN. verdict_scope: n/a (design note, no
   measurement).
3. **Rate framing**: NCA weights are COUNTED (learned, rule-118); seeds are tiny; iteration is
   FREE decode-time compute (inflate.py is a free interpreter, 30-min budget) — the exact
   compile-the-generator shape our rate law favors. #146's measured wall stands until re-measured.
4. **Texture trunk (#395) adjacency**: their texture synthesis on meshes = band-limited
   self-organizing texture from a local rule — an alternative T-generator for W=(G,ξ,T) IF the
   texture trunk's band-designed MLP underperforms; queue behind the texture A/B, not before.

**Disposition: LENS + reformulation-candidate banked for the AMBER/P10 reactivation queue + a
T-generator alternative behind #395's A/B. NOT an immediate lever** (the live chain and terminal
solves outrank; representation-side swaps need exact-gated arms).

Triality: papers-checked class — no lever, no measurement; DSL/equations N/A (stated).
