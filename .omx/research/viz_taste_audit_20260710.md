# Visual-taste audit + re-execution pass — 2026-07-10

**Subagent:** viz-taste-audit · **Governing bar (operator, 7 refinements folded):** journal/lab-grade tone ·
real-math/real-video content use · 3Blue1Brown storytelling · whitespace/structure · CRAFT/PROPORTION
("no arbitrary AI toy SVG") · dashboard = the existing high-water mark to port from · notebook must read
as a beautiful DOCUMENT to anonymous visitors · video re-make = abstraction→real spine, voice replaced or cut.
**Hard rules honored:** zero data/claim/measured-number changes; axis tags preserved; no bundle-runtime break
(reseal verified end-to-end); no #343 dashboard file touched; live-arm memory respected (no heavy renders).

Evidence dir: `.omx/research/viz_taste_audit_evidence_20260710/` — *.png gitignored; canonical copies live in
the session scratchpad (`viz/` subdir) and are enumerated below by basename so they can be re-rendered from
the committed generators at any time.

---

## 1. Inventory (what exists, where, and its generator)

| # | Artifact | Generator | Status this session |
|---|---|---|---|
| 1 | Published marimo notebook (42 cells), molab URL on `adpena/witness-machine` `notebooks/witness_machine_v12.py` | `src/molab_witness_machine/{v12_visuals,v12_real_evidence,v12_temporal,v12_copy}.py` + sealed bundle | **RE-EXECUTED (5 figures) + hide_code all 42 cells + bundle resealed v1.2.1 + pushed** |
| 2 | Temporal-transport plate (real pair 392→393) | sealed artifact `artifacts/v12_public/v12_temporal_transport_display.svg` (+npz) | critiqued; 1-panel fix punch-listed (artifact regen = manifest relock) |
| 3 | Paper figures ×3 `docs/paper/figures/*.png` (leaderboard, decomposition, step curves) | **ORPHANED — no generating script in repo** (grep verified); `MISSING.md` confirms unwired | critiqued; punch-listed (needs generator + shared style module) |
| 4 | Witness showcase `tools/build_witness_showcase.py` + `witness_showcase_app.html` | pact tools | not visually rendered (budget); punch-listed |
| 5 | 6-panel comparison GIF/MP4 convention (`src/tac/visualization/comma_gif.py` etc.) | pact | exempt-canonical per operator; consistency pass punch-listed |
| 6 | Dashboard (ORACLE/WITNESS/FLOW/WHY-HOW tabs), `tools/dashboard_server.py` + `oracle_dashboard_panels.py` + `whyhow_deepmath_panels.py` + WebGPU clients | pact | **mined as source material** (see §4); #343 files untouched |
| 7 | Narrated video (1080p30 release candidate 41.7 MB + mobile + animatic + silent) `~/Downloads/molab_witness_machine_v75_v8/artifacts/video/v12/` | `video/v12/build_animatic.py` (46 K), narration = **macOS `say`, voice "Samantha", 120 wpm** (`beat_manifest.json`) | audited; re-make PLAN below (operator-sanctioned plan-before-render) |
| 8 | Paper-figure SVG locks `artifacts/figures/v12/01..12_*.svg` (witness-machine release tree) | `v12_release.build_v12_figure_lock` | inherit the module fixes on next lock rebuild |

## 2. Top-5 worst offenders (before-state, concrete)

1. **Anonymous-visitor page = walls of code.** 0/42 cells had `hide_code`; molab preview showed raw source
   before any narrative. *Fix landed:* `@app.cell(hide_code=True)` ×42; preview now collapses to output-first
   document (residual: ~10 no-output bootstrap/picker stubs still precede the hero — punch-list P1).
2. **Two different SegNet palettes in one document.** Real-evidence plate used Okabe-Ito class colors while the
   temporal plate used comma10k-canonical — the same physical object (frozen SegNet argmax) encoded two ways.
   *Fix landed:* comma10k canonical everywhere (Road 64,32,32 · Lane 255,0,0 · Undrivable 128,128,96 ·
   Movable 0,255,102 · MyCar 204,0,255), class order = frozen-scorer canonical order.
3. **Red-vs-green paired heatmaps** (STAC sensitivity vs allocation): colorblind-hostile + arbitrary hue switch
   for coupled quantities; 5 px uniform crayon strokes. *Fix landed:* coral(=debt)/gold(=budget) semantic pairing,
   stroke hierarchy 3/2.25/1.25, annotation ink quieted.
4. **Candy-block MERGE/DIFF/CORRECT chips + 4-hue bar chart with unexplained dots** — the "goofy flowchart"
   register. *Fix landed:* quiet typographic stages (semantic top-rules, letterspaced 650-weight text, hairline
   arrows); shadow-price bars single ink hue with the admitted carrier as the ONLY color accent, dots removed.
5. **Buried headline stat.** The figure that carries the paper's core claim (sensitivity concentrates on the
   separatrix) hid its numbers in an SVG `<desc>`: boundary = 2.7 % of pixels, 99.9 % of margin-normalized
   sensitivity mass. *Fix landed:* in-figure tabular-numeral stat strip ("2.7 % of pixels are boundary ·
   99.9 % of sensitivity mass sits there"), honest colorbar label ("low · log display scale"), crisp
   (`image-rendering:pixelated`) label panel, card radius 18→8. Numbers unchanged (read from the locked manifest).

## 3. Per-artifact five-axis grades (TONE / CONTENT-USE / STORY / STRUCTURE / CRAFT)

| Artifact | TONE | CONTENT | STORY | STRUCT | CRAFT | One line |
|---|---|---|---|---|---|---|
| Notebook page (visitor view), before | pass | fail | fail | fail | fail | prose is serious, but code walls buried it; richest real fields under-shown |
| Notebook page, after this pass | pass | part | pass | part | pass | document-first; residual stub rows + slider bleed remain |
| Real-evidence plate, before→after | pass | part→pass | fail→pass | pass | fail→pass | stat now told; one palette; crisp panels |
| STAC toy panels, before→after | pass | pass (TOY, honestly labeled) | pass | pass | fail→pass | red/green killed; stroke system |
| Carrier graph + pipeline, before→after | fail→pass | part | pass | pass | fail→pass | chips de-goofed; stroke hierarchy |
| Shadow-price bars, before→after | pass | pass | pass | pass | fail→pass | one hue, one accent |
| Temporal-transport plate | pass | pass (real pair) | pass | pass | part | receiver panel ink inverted (97 %-valid field saturates; failures should carry the color) — punch-list |
| Morse–Smale / Laguerre / SDF scenes | pass | part (toy-first is correct as the abstraction beat; missing the real-data twin) | part | pass | part→pass (stroke/fill quieted) | need the abstraction→real pairing |
| Paper figures ×3 | pass | fail (bare bars/curves; no video, no geometry) | fail | part | fail | serif default-matplotlib; ORPHANED generators |
| Dashboard tabs | pass | **the high-water mark** | pass | part | part | see §4; critique-only per #343 |
| Video (release 1080p30) | fail | part (real frames used, but decorative checkerboard instead of the true field) | fail | part | fail | Samantha TTS; ALL-CAPS chunk; text collisions (T3 bar overlaps caption at 02/08) |

## 4. Dashboard mining (the operator's named high-water mark)

Portable "cool examples" identified (generators, all pact-side, read-only for me):
- **ORACLE tab** (`oracle_dashboard_panels.py`): "the detector I built, and the world it reads" — frozen scorer +
  openpilot physical priors (lane band → d_seg, ego-ξ screw → d_pose) + SegNet detectability field.
- **WHY/HOW tab** (`whyhow_deepmath_panels.py` + `dashboard_whyhow_client.js` WebGPU plates): co-registered scalar
  fields — ρ_seg margin / real S-UNIWARD cost / separatrix sensitivity on ONE frame. This is the geometry language
  the notebook should speak.
- **FLOW tab** (`dashboard_flow_sequence.py`): full-video witness render + margin-fragility layer per frame.
- **WITNESS tab**: live 6-panel comma10k render from the EMA checkpoint.

**Ported this session (within sealed-bundle constraints):** the WHY/HOW field-plate language into the notebook's
real-evidence figure — the bundle's locked pair-196 fields (`argmax_u8`, `boundary_u8`, `margin_f32`,
`flip_risk_f32` at 384×512) are exactly the WHY/HOW plate ingredients, so the re-executed plate (comma10k
partition + log-field panels + concentration stat) is the dashboard demonstration expressed from bundle-only
data, with zero reseal risk to the evidence chain (modules swapped, data untouched).
**Not portable without adding data to the bundle:** a real RGB/luma frame (bundle has no photographic substrate),
level-set evolution across epochs, FLOW sequences. → punch-list P2 with the reseal recipe now proven (v1.2.1).

## 5. 3b1b operational grammar (studied from primary sources this session)

Extracted from `3b1b/manim` `manimlib/default_config.yml` + `constants.py` (exact values, traceable):
1. Ground is **#333333 dark grey**, never pure black, never white-default.
2. The palette is **families with five lightness steps** (BLUE_E #1C758A → BLUE_A #C7E9F1): hierarchy comes from
   lightness within one hue, not from adding hues.
3. **Yellow (#FFFF00 family) is the single emphasis color** — used only at the current point of attention.
4. Browns/tans (#736357, #8B4513, #CD853F) are the warm counterweight, not a data channel.
5. One default stroke (4.0 at 1080p) with deliberate ratios for sub/super-ordinate strokes — never uniform outlines
   on everything.
6. Text enters sparsely and large (unit-height font 144); equations build term-by-term, synced to the object each
   term describes.
7. A scene STATES ITS QUESTION visually before answering (the setup IS a picture, not a caption).
8. Motion only when the mathematics moves — interpolate the actual quantity (the boundary, the field), never
   decorative transitions.
9. Long holds on key frames; pacing follows comprehension, not rhythm.
10. Abstraction first, then the same construction on the real object (the recognition beat).
11. Color MEANS the same thing across a whole video/series (his blue=object, yellow=highlight discipline).
12. Nothing touches frame edges; generous dark margins carry the composition.
Applied translation for this project (kept the notebook's light instrument identity rather than wholesale dark
conversion — fidelity + budget): one hue = one meaning (cyan=structure/ours · gold=budget/rate · coral=debt/failure ·
mint=valid/seg-ok · comma10k = partition classes, everywhere), lightness-step hierarchy, single accent per figure,
stroke ratios 3.5/2.5/2/1.25.

## 6. What landed (files + commits)

**witness-machine repo (pushed to main, `5cdf168`; release `v1.2.1` created):**
- `notebooks/witness_machine_v12.py` — `hide_code=True` ×42 cells; bundle pins → v1.2.1
  (bytes 3 705 237, sha256 e8494c72…5472). URL/path/branch unchanged.
- `src/molab_witness_machine/v12_visuals.py` — STAC coral/gold + stroke system; carrier-graph stroke hierarchy
  + typographic pipeline stages; shadow-price single-hue+accent; Laguerre stroke/fill/site proportions.
- `src/molab_witness_machine/v12_real_evidence.py` — comma10k-canonical palette (scorer class order), stat-strip
  headline (values read from locked manifest — no new measurement), pixelated label panel, viewBox 402→448,
  radius 18→8.
- `src/molab_witness_machine/v12_copy.py` — colorbar labels name the scale (EN+ES).
- `tests/test_molab_runtime_hotfix.py` — decorator matcher accepts `@app.cell(hide_code=True)`; 7/7 pass.
- `notebooks/__marimo__/session/witness_machine_v12.py.json` — refreshed; contains the new figure outputs
  (string-verified: stat-strip text present).
- Bundle reseal: deterministic zip, same member order/timestamps as rc2, 3 modules swapped, in-bundle
  RELEASE_MANIFEST entries updated for swapped members only; **download chain verified end-to-end**
  (fresh cache → HTTPS download → bytes+sha validation → extraction → required-files check → swapped-content
  assertion) and **isolated molab-faithful export** (notebook alone + fresh cache) produced 0 tracebacks.

**pact repo:** this memo (+ checkpoint rows). No .py landed pact-side this session (see §8 honesty).

## 7. Before/after evidence (scratchpad `viz/`, re-renderable from committed generators)

- Page: `wm_top.png` / `wm_s1..s6.png` (before) → `after_2100/3350/8730/10660.png` (after) →
  `public_visitor_top.png` (live molab post-push).
- Isolated craft delta (same data, same idea): `wm_s1.png` vs `after_2100.png` (STAC pair) and the
  real-evidence panel in `wm_s2.png` vs `after_3350.png` (palette + stat strip + labels only).
- Video stills: `vid_15/75/150.png`.

## 8. Honest limits of this session

- **Public preview lag/scroll:** molab's read-only preview iframe is cross-origin; I verified the new outputs via
  the committed session snapshot + local export of the exact pushed notebook, and the hero/first viewport live,
  but could not screenshot the deep-page live figures. Re-check once molab cache turns over.
- **"cells with outputs" count in snapshot** read 0 under my grep key while the content strings are present —
  the snapshot format nests outputs differently; content verification is the one I trust.
- **Voice aesthetics** not judgeable by me even after re-synthesis; any re-voice needs operator listen-check.
- I could not visually render: witness showcase suite, pact paper-figure regeneration, dashboard live tabs.

## 9. Ranked punch-list (not fixed, with why)

**P1 — notebook first-viewport stubs (visitor experience).** ~10 no-output cells (imports/bootstrap/pickers)
render as collapsed stubs before the hero. Fix: reorder cells (marimo is order-independent reactively) so the
hero mo.Html cell is file-first, or merge picker defs into consuming cells. Low risk, ~1 h, needs an export+
snapshot cycle. Not done: session budget after reseal chain.
**P2 — real-frame ports into the notebook** (the recognition beat): add one luma frame (≤80 KB PNG u8) + a
per-epoch boundary-evolution strip (from cached run artifacts, NOT n600 recompute) to the bundle; reseal per the
now-proven v1.2.1 recipe. Then: annulus-on-real-road hero; abstraction→real twin for the Laguerre and Morse–Smale
scenes. Needs pact-side cached artifacts selection + operator eyes on bundle growth.
**P3 — temporal plate receiver-panel ink inversion** (valid=quiet neutral, holes/collisions=the only saturation):
requires regenerating the sealed SVG artifact + its manifest relock via `v12_temporal.py` → do together with P2's
reseal to avoid two seal cycles.
**P4 — pact shared style module `src/tac/viz_style.py`:** semantic tokens (cyan/gold/coral/mint + comma10k classes),
modular scale (base 8 px, ratio 1.25: tick 10/label 12.5/title 20/hero 25), stroke ratios (2.5/1.25/0.5 pt),
rcParams (constrained_layout, ≤6 round ticks, tabular-nums, DPI ≥200, no default 6.4×4.8 — aspect chosen per
figure), colormaps: viridis/magma family only for fields. Every generator in `tools/render_*` imports it.
Not landed: .py landing requires the full verify-landing chain (review gate, serializer) — out of budget after
the notebook chain took priority per refinements #3/#5.
**P5 — paper figures:** write `tools/render_paper_figures.py` (data transcribed 1:1 from the existing PNGs — all
historical numbers preserved) through P4's module; sans-serif, one hue + accent, matched panel scales in
step-curves (right panel should state both panels' differing y-scales in-frame), drop "(x10⁻³)" for scientific
offset notation, wire into `docs/paper/` markdown per `MISSING.md`.
**P6 — video re-make** (plan for cheap course-correct BEFORE render):
  - Voice: **recommendation = Option 2, no voice** — typographic narration at beat boundaries (classier, zero
    account risk, removes the Samantha problem entirely). If voice is wanted: Kokoro-82M via mlx-audio (local,
    no signup); flag sample for operator listen before any ship.
  - Spine (abstraction→real, per beat): (1) two power cells + separatrix toy → real pair-196 partition + its
    boundary; (2) 1-D level set crossing zero → φ zero-set on the real frame; (3) toy argmax margin →
    log-margin field + 99.9 %/2.7 % concentration stat; (4) single screw motion on a box → real ego-ξ between
    frames 392→393 (bundle's dense-flow field); (5) budget waterfill toy → STAC allocation on the real
    sensitivity field (replacing the current decorative checkerboard overlay with the TRUE field);
    (6) close on the score law, term by term, each term lighting its visual object.
  - Craft: #333333-family ground, one emphasis color, no ALL-CAPS chunk titles, fix the 02/08 text collision,
    render figure layers from the SAME generators as the notebook (single source of taste).
**P7 — dashboard punch-list (critique-only, #343 owns files):** unify its seg palette with comma10k canon where
any tab differs; carry the notebook's stroke/typography tokens into `dashboard_*_client.js` once P4 exists;
export ORACLE/WHY-HOW plates as the notebook's static twins (they already read at the right register).
**P8 — `tools/mdl_visualize_bytemap.py` uses `tab10`** — swap to semantic/uniform palette when P4 lands.

## 10. Fidelity + honesty statement

No measured number, claim, axis tag, evidence manifest value, or locked artifact was altered. The stat strip
surfaces numbers already present in the locked manifest. Bundle data members are byte-identical to v1.2.0-rc2;
only the three renderer/copy modules and their manifest rows changed. The paper-reconstruction spine (cells,
claims, tables, TOY/ADVISORY/EMPIRICAL labels) is untouched.
