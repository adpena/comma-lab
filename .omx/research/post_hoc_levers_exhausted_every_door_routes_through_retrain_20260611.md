# CONVERGENT VERDICT: every $0 post-hoc lever on the current frontier is exhausted — the only remaining pointer-mover is a paid RETRAIN (2026-06-11)

**Authority:** two independent DAG-first $0 exact-row hunts (torch-CPU advisory, NO MPS, no dispatch),
both returning honest negatives that CONVERGE. Frontier UNMOVED **0.19109982419 [contest-CPU], 177,169 B,
sha b46897267d**. Recomputed exactly from components: seg 0.055978 (29.3%) + pose 0.017152 (9.0%) + **rate
0.117970 (61.7%)**. This is a decisive strategic verdict, not a pointer move. No signal lost — every lever
proven banked or correctly fail-closed.

## Hunt 1 — score-aware per-tensor weight RE-QUANT (#69): EXHAUSTED (whole-tensor)

Every re-quant candidate is STRICTLY WORSE. Best = crush4_int6: −18,868 B but S=0.35130 (+0.16020 the WRONG
way); the gentlest crush pays a distortion penalty **13.8× the rate saving**. Mechanism: the HNeRV decoder
is a single-video MEMORIZER (6 PixelShuffle/sin stages); quant error compounds multiplicatively, flipping
SegNet argmax + perturbing pose. Every surviving int8 bit is score-load-bearing — **no recon-only fat**.
The 62%-of-score rate term is NOT reachable by post-hoc weight coarsening at tensor granularity.
Reactivation (un-attempted on this frontier): (1) per-CHANNEL re-quant (weaker per-output-channel
compounding); (2) **score-aware QAT fine-tune** — re-quant then fine-tune surviving levels against
SegNet/PoseNet so the network RELOCATES the decision boundary into the coarser cell. **(2) is a RETRAIN.**

## Hunt 2 — waterfiller atom inventory (#30 pre-work) + orphan recovery (#48/#72/#54): NO ready candidate

The current frontier is a **PROCEDURAL HNeRV** — member-x = decoder 162,127 B (90.9%) + latent 15,387 B
(8.6%) + selector ~879 B (0.5%). **It stores NO camera-frame pixels.** Consequences:
- **Certified-invisible / null-space atoms (#47, S12): 0 addressable bytes** — they operate in the
  camera-pixel domain that this archive does not contain. INAPPLICABLE on a procedural vehicle.
- **Rate axis:** at the 7.999-bits/byte entropy floor (R1/R2 recode = the last win, −0.000883, already
  banked into b46897267d). CLOSED.
- **Seg axis (#72 Lever D):** the margin-conditional coder BEATS the 1.27 B/flip break-even (0.856 B/flip),
  but every frame-1 correction creates MORE bad flips than it fixes — the frontier residual is
  **salt-and-pepper (95% single-pixel flip components)**, so receptive-field collateral eats every
  correction (0/24 net-positive; −2,356 at full strength). Binding constraint = collateral, not rate.
- **Pose axis (#48 selector, #54 corrector):** SATURATED — the FEC6 selector is already per-pair
  pose-optimal (0/42 improvable); the cross-pair waterfiller admits 0.
- **Orphan recovery:** #48 R3 (−2.58e-6) is exact-evaled + BANKED; R1+/R2 were built on the WRONG-GT macOS
  table R3 proved contaminated (do NOT eval — burning $ to confirm a GT artifact). #72/#54 correctly admit
  0 at the local kill-gate. **Zero ready un-run byte-closed candidates beat the frontier.**

## The convergence (the structural conclusion)

Both hunts, from different levers, route the only remaining door to the SAME place:
- Hunt 1's live reactivation = score-aware QAT = **a retrain**.
- Hunt 2's #30 waterfiller actuator (`allocate_seg_regions` + `compose_water_level_allocation` #54 +
  margin-conditional coder #72) is **BUILT but STARVED** — it funds corrections only when fed a base whose
  seg residual is **contiguous multi-pixel patches**, and a contiguous-residual base is a **training
  product**, not a byte-transform.

**⟹ Every remaining road to sub-frontier runs through a paid RETRAIN.** The $0 post-hoc toolbox is empty:
re-quant (worse), entropy recode (floored), invisible atoms (0 bytes on procedural), seg-repair (collateral
floor), pose selector (saturated), smaller-basis (capacity-walled, refuted earlier today).

## The decision (operator-gated — irreversible paid spend)

The retrain is NOT the refuted smaller-basis (Cool-Chic walls d_seg ~0.014) and NOT the naive base_ch=20
n600 (capacity-walled). The viable retrain targets, all paid, all uncertain (must beat a tuned frontier):
1. **Score-aware QAT on the frontier decoder** — re-quant to int4/int5 + fine-tune surviving levels through
   SegNet/PoseNet (eval-roundtrip + EMA warmup) to relocate the boundary into the coarser cell → a RATE win
   the post-hoc re-quant could not get. Smallest, most-targeted bet; reuses the frontier decoder.
2. **Contiguous-residual base retrain** — train a base whose seg residual is contiguous patches (not
   salt-and-pepper) so the BUILT #54/#72 waterfiller actuator can fund corrections → unblocks #30 as a
   pointer-mover.
3. **Frontier-class beat-PR95 retrain (#90 reframe)** — a frontier-class decoder that beats PR95's
   d_seg/d_pose at ≤ frontier bytes (distortion win at equal rate, since frontier-class ≈ frontier bytes).

The $100 Modal grant exists to BUY exact rows. With the $0 toolbox exhausted, a paid retrain is the
RIGHT default — but the TARGET (1/2/3) is an operator decision, and (1) score-aware QAT is the cheapest,
most-targeted, highest-information first spend (it reuses the frontier decoder + directly attacks the 62%
rate term via the one lever — boundary relocation — that the post-hoc re-quant proved is the missing piece).

## Bottom line (non-sycophantic)

The frontier is genuinely hard now: UNMOVED at 0.19110, and this cluster proved — across re-quant,
recode, invisible atoms, seg-repair, pose selector, orphan recovery, and the smaller-basis capacity test —
that **no $0 post-hoc lever moves it.** That is not a stall; it is a sharpened map: every door is a paid
retrain, and the cheapest, most-targeted, most-informative first spend is **score-aware QAT on the frontier
decoder** (boundary relocation = the rate lever post-hoc re-quant proved is missing). The honest next step
is an operator decision on which retrain target gets the $100 — the $0 work that precedes it is done.
