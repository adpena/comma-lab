# ddm_rb1 — FREE deterministic post-render boundary treatment against the measured 86 % one-pixel jitter residual ($0 falsifier at n600; seal if it pays; charter, MAIN 2026-09-11; operator full-authority GO)

## The opening
MEASURED (memory `residual_seg_debt_is_renderer_boundary_jitter_at_correct_tokens_20260908`, sha 9c223361163636b6…): 86 % of the pointer's
remaining d_seg is renderer boundary JITTER at CORRECT tokens — one-pixel placement error at class boundaries after the render (Lane 40×
over-represented). rw1 closed REPAIRS of the renderer's int4 code grid (no repair survives n600). Nobody has tried a FREE, deterministic,
generic post-render treatment inside `inflate.py` — rule 118 free compute — that places the boundary where the scorer's argmax wants it:
sub-pixel boundary snapping from the token plane's known class edges, anti-aliasing consistent with the scorer's downsample, guided
filtering of the rendered RGB by the token-plane edge map, or a morphological consistency pass. d_seg is 0.00010345 (term 0.010345);
removing a quarter of the jitter is −2.2e-3 S (110 bars) at ZERO bytes. The scorer-side physics: SegNet reads RGB at 512×384 after
bilinear resize (the same D as PoseNet) — treat every candidate through the real R operator.

## Deliverable
1. Instrument the residual on the SHIPPED bytes at n600: render with the shipped receiver (move-44 tree, read-only copy), score through the
   frozen CPU scorer [macOS-CPU advisory], and classify every argmax disagreement vs GT by (token correct?, distance to nearest class edge in
   the token plane, class pair) — reproduce the 86 % / 1-px figure on THIS field (falsifier of the instrument).
2. Three free treatments, each ORIGINAL to this vehicle with an OSS anchor (guided filter; edge-aware anti-aliasing / SSAA at the class edge;
   token-plane-driven sub-pixel boundary displacement à la signed-distance rasterization), each applied INSIDE the render path (deterministic,
   integer/fp32-stable, identical on T4), each measured at n600: d_seg, d_pose (both — the renderer seg↔pose coupling law), decode wall-clock
   delta. Report ΔS per treatment and the best composition; keep every render.
3. If the best treatment beats the −2e-5 bar at unchanged bytes and pose: it is a RECEIVER CHANGE → produce the receiver delta, the
   candidate tree, twin archive encodes (bytes unchanged except any header), full cold n600 public parse-back, manifest from outside the tree,
   literal census (rule 118: no fitted constant may enter the code — a treatment parameter must be a generic constant or DERIVED from the
   token plane at decode time; if it is fitted to this video it is COUNTED and must go in the archive), and hand MAIN the first-measurement
   intent inputs (timing-risk receipt with the measured wall-clock delta). Do NOT fire.
4. Memo `.omx/research/ddm_rb1_free_post_render_boundary_treatment_20260911.md`: the residual instrument, the three treatments' n600
   table, the composition, the intent inputs (or the measured closure). Serializer commits (two review passes per .py); rc 17 is NOT a stop.
   Checkpoint `ddm_rb1`; COMPLETE.

## Boundaries
No Modal, no scorer weights in the receiver (strict-scorer rule), no fitted per-video constants in free code (rule 118), never edit
`upstream/`, the PR tree, sealed trees; n600 only; every render retained with sha; heavy steps through the launcher; do not touch
obx2/pc3/ntb1/mxo1/gp1 directories.

## OPTIMAL FORM
- Reference form: the shipped render path + the frozen scorer through the real R operator at n600 as the ONLY verdict path; a treatment
  measured on a subset is SCOPE (declared, no verdict); a proxy scorer or a different resize is MECHANISM (toy-bracketed).
- Provenance pins (sha256 prefixes): jitter memory (record sha); move-44 packet memo f7638e1e171e0d19…; rw1 memo (`renderer_int4_code_grid_smallest_action_breaks_240_cells_20260909`,
  record sha); pointer commit 99625f32f / archive 04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e.

## Prior negatives accounted (operator 2026-08-15)
- rw1: grid REPAIRS break at n600 (240 = screen extrapolation) — measure at n600 only; a render-time filter is a different family; say why.
- rf1/ft1 (seg↔pose coupling 170–220): seg-only treatments are unpayable — measure pose for every treatment.
- move 41 (rule 118): a fitted scalar in receiver code is content — every treatment constant is generic or derived at decode time.
- tc4: a receiver change needs a measured decode wall-clock — report the delta per treatment.
- ANE/fp16 drift law: keep the treatment in fp32/integer; no MPS/ANE authority.

Final message: the residual reproduction, the treatment table (d_seg, d_pose, ΔS, wall-clock), the composition, the intent inputs or the
closure, and the frontier line `composition S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600] (move 44)`.

<!-- # FORMALIZATION_PENDING: producer charter; the treatment law lands in the equations leg with the exact row -->
