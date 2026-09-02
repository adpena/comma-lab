# WW1 — walls that weren't walls (operator 2026-09-02: "What other walls weren't walls conclusively")

Date: 2026-09-02. Owner: MAIN. Sources: the negative-audit line (na2 · na4 · na10 · na11 ·
rv1 · ny1 · ch1 confound hunts) + task-ledger receipts cited per row. Three honest columns —
dissolved, held, never-settled. The DISSOLUTION MECHANISMS recur; they are the taxonomy.

## 1. WALLS THAT DISSOLVED — with the mechanism that faked them

| Claimed wall | Dissolution | Mechanism class |
|---|---|---|
| Capstone d_seg plateau 0.505 ("true wall") | Re-diagnosed 4× — EMA-shadow-lag bug (#85), spike-guard median-freeze (frozen run certified "converging" — the 18-confound hunt), under-training, broken curriculum | BUG + FROZEN-INSTRUMENT |
| B1 HiNeRV "doesn't learn" (d_seg 0.50 flat, #34) | Inert score-aware loop — gradient pathology, cured by #76 | BUG |
| The seg training "wall" | **92.7% CONFIGURATION** (#1091): 81.19% of the LR budget went to the worst-aligned objective (#1089), scale-invariant split; aiming budget at the aligned objective closes 13.6× | CONFIG |
| Fixed-β hosc divergence ("activation family dead") | Pre-FINER++-cure toys; #310 bias-init is the published fix; verdicts were implementation-level | TOY + MISSING-CURE |
| mod-dim capacity wall | Seed-compose island-gradient STARVATION (#300) — a wiring defect refuted the capacity reading | BUG |
| "Physics floor" on recoverable generators | REFUTED 3-0 (#213 header) | WRONG-ATTRIBUTION |
| Label-noise floor ("residual d_seg is label noise", #141/#169) | td1/rt1 (#1075): **95% of seg error is MANUFACTURED** by the render path, not label error; mst1 (#1211): 78.71% appears at the native render and R+uint8 are net REPAIRERS | WRONG-ATTRIBUTION |
| The realization "wall" (edits die in realization) | #897: 96.6% of realization flips CURED (88→3) — solver-path bugs; qs3/#1037 inverted the mechanism (loss is COLLATERAL, 97.4% of edits realize) | BUG + WRONG-ATTRIBUTION |
| Pose "collapse" (d_pose 2.67–12.66) | Was the amortized-luma-CARRIER composition — a suboptimal sibling, never the stored-target/pose family (canonicalized in CLAUDE.md) | WRONG-OBJECT |
| Witness post-hoc pose "MEASURED DEAD" | na2 label correction: n8/n24 PREFIX ≈ 0.5 effective samples, anti-conservative 2.54–4.21× on pose — downgraded to standing-verdict-on-weak-evidence | TOY-SCALE + PREFIX-BIAS |
| int4/int5 PTQ "collapse → KILL" | The canonical premature-kill anchor — naive PTQ without per-channel/LSQ/outlier handling; #147 re-test at best-shot | NAIVE-FIRST-PASS |
| Cool-Chic d_seg/pose wall (#115) | Under-powered by its own label ("still descending, NOT a basis limit"), n48, never re-run — superseded by qbr1's proper discriminator | TOY-SCALE |
| "Sub-0.12 unreachable by distortion alone" (binding banner) | STALE ARITHMETIC (#1356): premise expired 453.6 B ago; zero-distortion corner clears with 216.35 B margin | STALE-CONSTANT |
| The address-tax "law" (every closure = address tax) | rt3 (#1342): POST-HOC — only 1 of 5 closures actually measured an address; route table was never empty | PATTERN-OVERFIT |
| W96 renderer family "closed" | #1225: closed from TOY rows — SVD r32 dead unconditionally, W96 itself NOT closed → the w96/qbr1 line exists because this wall was retracted | TOY-SCALE |
| Window cost "wall" (4.9× too expensive to probe, #1087) | A 50-step end-to-end smoke became the campaign cost model — measurement artifact, not economics | STALE-CONSTANT |
| Rate-axis prefix bias (assumed pose-like 2.5–4.2×) | na4: measured 0.989×–1.030×, sign-variable — the wall didn't generalize across axes | AXIS-TRANSFER |
| τ 8,942× seg/pose coupling (#1253) | RETRACTED — seg lives on boundaries, pose in interiors, near-disjoint supports | WRONG-OBJECT |
| "Localization is the tax" (address floor read as distortion ceiling) | m98 SEQUEL: LOST — refused ×2 at 686×; removes a byte FLOOR, not a distortion CEILING; n≥8 address-free members all closed | CATEGORY-ERROR |
| PR135 F26 "truncated-still-accepting" (resume headroom) | Premise refuted at source (#1033): pass 8 = 0/595 accepts, CONVERGED — conflated F23's improving pass | READING-SEMANTICS |

## 2. WALLS THAT HELD under proper re-measurement (the honest column)

- **Round-trip accuracy intercept ~140,477 B** — TWO independent arms, opposite directions,
  same clause; above it no token accuracy reaches sub-0.12 (memory row, 08-31).
- **Pose absolute budget ≤1.25e-4** (∂S/∂d_pose 626.5, m110) — survived the same sweep that
  killed "localization is the tax"; the ratio ladder 33.7–922× is like-for-like.
- **Sharp-optimum law** — the HPAC field+model joint local optimum, measured by FIVE arms in
  every perturbation direction (#1214).
- **gf1 generator capacity ceiling 5.09×** — target-independent (#1334), gap decomposed
  (#1337), ordering exhausted.
- **Lane carriage floors** — ltg1 233,262 B (6.47×) and blp1 60,191 B weight-only (1.67×):
  measured floors on four representations (#1365/#1366).
- **br2/qxr1 born-object distortion refusal** — d_seg 0.1708 / d_pose 115.84, measured at
  n600 and confirmed by construction — though see §3: WHY is the live question.

## 3. NEVER CONCLUSIVELY SETTLED — each with its live discriminator

1. **Born-object distortion: capacity or optimization?** — qbz1 verbatim: no capacity fork
   was ever claimed. Discriminator = **qbr1, burning NOW** (cell 1/6). This is #115's
   question reborn at proper power.
2. **The seg asymptote-above-init (#1088)** at 5× window — refuted my truncation hypothesis
   but the CONFIG cure (#1091's 13.6×) was never run to endpoint; #1089/#1090/#1091 still
   pending. The CE1 exact loss (built since) is the missing instrument — partially subsumed
   by qbr1's aligned-loss treatment arm.
3. **W96 aligned-loss hypothesis** — w96a honest-blocked, w96b built the exact law; seeds
   rode #1304's storage-gated fire; the qbt2b→qbr1 line carries it. Not yet a verdict.
4. **Seed variance (#1251)** — ONE seed across every wd3 run ever; partially paid (two-seed
   OFF screen), fully paid only by qbr1's 3-seed design.
5. **Exchange-ratio noise floor (#1248)** — nobody has measured it; every ratio-based
   close inherits this caveat.

## 4. THE TAXONOMY (what fakes a wall here)

Six mechanisms produced every dissolved wall: **BUG** (inert loop, starvation, frozen
guard) · **TOY-SCALE/PREFIX** (n8/n24/n48 vs n600; pose 2.5–4.2× anti-conservative) ·
**CONFIG** (misallocated budget, wrong loss form) · **WRONG-OBJECT/ATTRIBUTION** (a sibling
implementation's failure billed to the family; manufactured error billed to labels) ·
**STALE-CONSTANT** (binding numbers that outlived their premise) · **PATTERN-OVERFIT**
(a law induced from closures that never measured its variable). The standing cures are
already law: #307 implementation-vs-paradigm, the n600 allergy, charter-time optimal form,
the m52 no-binary rule, re-derive-at-pointer-move, and the verdict-scope ladder.

## 5. Consequence for the live campaign

The three biggest CURRENT walls each carry a named discriminator: born-distortion → qbr1
(live) · trainer-config wall → CE1-aligned arms (inside qbr1's treatment) · the
representation byte-wall → the accuracy-intercept + sharp-optimum laws (HELD — these are
the real ones the object-change door must go around, not through).
