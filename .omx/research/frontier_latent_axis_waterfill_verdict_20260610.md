# Frontier LATENT-AXIS waterfill — the 15,387-byte latent blob + 607-byte sidecar (9.1% of the 0.19199 archive)

**Subagent:** `frontier_latent_axis_waterfill_20260610` · UTC 2026-06-10.
**Lane:** `lane_frontier_latent_axis_waterfill_20260610`.
**Mission (the decoder-axis verdict's named reactivation criterion #3):** the decoder-axis verdict
(`frontier_decoder_axis_waterfill_verdict_20260610.md`) proved the decoder distortion lever is
DEFER (coarsening kills d_seg ~10× the rate gain), and named the **latent axis** (15,387 B latent
+ 607 B sidecar = 9.1% of bytes, ~0.02% of per-byte-uniform |grad|) as the lowest-sensitivity
remaining surface. This memo attacks it.
**Axis discipline:** the per-pair/per-dim sensitivity map + RD probes are `[macOS-CPU advisory]`
candidate-generator priors — NOT score claims. The paired contest-CPU eval
(`upstream/evaluate.py --device cpu`, Linux-x86_64 Modal CPU container, 1:1 with the contest GHA
runner) is the score authority. The frontier remains **0.19198534 [contest-CPU]**.

---

## 1. The latent grammar, structurally (the attack surface)

The frontier `b7106c9b…` member-x source payload is `[decoder(162,127) | latent(15,387) |
sidecar(607)]`. The latent + sidecar are the canonical PR101-family grammar
(`tac.pr101_split_brotli_codec`, verified byte-identical decode of the frontier):

- **Latent blob (15,387 B):** `lzma.FORMAT_RAW + FILTER_LZMA1(dict=4096, lc=3, lp=0, pb=0)` of
  `mins[28×fp16] + scales[28×fp16] + delta[600×28 uint8]` (16,912 raw B). The deltas are
  **first-order temporal**: `q[i] = q[i-1] + ((delta[i]-128) mod 256)`, `q[0]=delta[0]`, dims
  permuted by `LATENT_DIM_ORDER`. Final latent = `q.float() * scales + mins`. 28-d per-pair, 600
  pairs, 2 frames decoded per pair (L19/L24/L25).
- **Sidecar (607 B):** canonical-Huffman-enum form (`SIDECAR_HUFF_ENUM_LEN`); per-pair single-dim
  additive corrections `latents[pair, dim] += SIDECAR_DELTAS_X100[code]/100`; `dim=255` = no-op
  (L26/L27/L31). **It corrects 597 of 600 pairs** (NOT a sparse high-value set) with tiny nudges
  (RMS 0.0069 = 0.76% of the latent magnitude 0.91; max |corr| 0.10).

### Two structural facts that bound every latent lever (the deep finding)
1. **The latent coding floor is EXHAUSTED.** The frontier's `dict4096/lc3/lp0/pb0` LZMA is optimal
   among ALL dict/filter/coder variants tested (dict 256–32768 × {lc/lp/pb combos}; brotli q11
   15,530 / bz2 16,324 / zlib 15,778 / LZMA2 extreme 15,398 — **every variant ≥ 15,387**). The
   delta stream is high-entropy (~7.0 bits/symbol; iid floor ≈ 14,772 B; LZMA at 15,387 − 112-B
   header = 15,275 B is within ~3.4% of the iid floor and exploits cross-pair context the floor
   ignores). **No lossless coding lever recovers material bytes.**
2. **Second-order temporal re-prediction is FALSIFIED.** Replacing the first-order delta with a
   second-order (linear-predictor) residual gives entropy 7.52 bits > the first-order 7.03 bits,
   improves 0/28 dims, and grows LZMA by 977 B. The dashcam latents are NOT smooth enough for
   second-order prediction; the authored first-order delta is optimal.

**=> The only latent levers are DISTORTION (lossy q-coarsening of the freest pairs/dims) and the
SIDECAR rent (drop the 607 B of tiny corrections if they no longer pay).**

---

## 2. The per-pair × per-dim sensitivity map (the bit-allocator prior)

`build_latent_sensitivity_map.py` → `latent_sensitivity_map.json`. Three FREE signals joined:

- **(A) Per-pair score-weighted sensitivity** from the master-gradient per-pair full600 ledger
  (`master_gradient_fec6_frontier_mlx_per_pair_full600_20260527.npy`, `(178,517 archive B, 600
  pairs, 3 axes)`): sum the latent compressed span over each pair. Headline: the latent carries
  **4.42% of total distortion sensitivity** (score-weighted: latent SW 36.55 vs decoder SW 790.53)
  — far MORE than the "0.02%" per-byte-uniform |grad| headline (which ignored the pose-heavy,
  score-weighted reality), but still ~2× more favorable per byte than the decoder. The **per-pair
  spread is enormous**: min SW 1.3e-4, median 3.77e-2, max 3.71e-1 (2,850× range). Freest pairs:
  `[219, 177, 260, 31, 51, 22, 109, 564, 345, 182]`. Hottest: `[283, 78, 88, 121, 508, 6, 340,
  275, 354, 159]`.
- **(B) Per-dim frame-MSE Jacobian** from the ACTUAL frontier decoder (perturb each latent dim,
  measure ‖d(rendered RGB)/d(dim)‖ × per-dim scale = the 1-LSB-delta-step distortion cost). The
  per-dim spread is NARROW (LSB-cost 0.27–0.60, only 2.2×): freest dims `[20, 14, 26, 10, 1, 15,
  5, 6, 25]`; hottest `[2, 11, 9, 21, 16, 7, 8, 17, 13]`. The per-dim lever is weak relative to
  the per-pair lever.

### Sensitivity-map CORRECTION of the prompt's protected-pair list
The prompt named pairs **510–522, 133, 177–178** as protect targets. The ledger authority
**contradicts this**: pair **177 is the FREEST pair** (0.2 percentile, SW 9.5e-4 — NOT protected);
pairs **512/514/522** are LOW (10th/12th/27th percentile). Only **517/519/521** (≈90th pct), **178**
(81st), **133** (65th) are genuinely hot. The candidate generator protects the ledger-verified hot
pairs (`HOT_PAIRS` = top-decile by ledger SW ∪ {517,519,521,178,133}), not the static list. This is
the per-pair-allocation correction THE LAW + the ledger demand.

---

## 3. The candidates (sensitivity-aware; byte-closed; decoder byte-identical)

`build_latent_candidates.py` reuses the canonical engine end-to-end (NO duplicative code):
`feca_selector_reparameterize.{split,join}_fp11_member` + `pr101_split_brotli_codec` latent
decode/encode. **Decoder + selector + dqs1 byte-identical to the frontier** (proven via re-split
asserts) — this is structurally a latent-axis move, orthogonal to the decoder-axis verdict.

| cand | mechanism | latent B | sidecar B | archive B | archive saved | rate ΔS (measured) |
|---|---|---:|---:|---:|---:|---:|
| L1 | sidecar drop (597 tiny corrections lost) | 15,387 | 0 | 177,886 | −607 | −4.04e-4 |
| L2 | per-pair coarsen freest 70% step4 | 14,758 | 607 | 177,864 | −629 | −4.19e-4 |
| L3 | per-dim coarsen freest dims step3 | 15,193 | 607 | 178,299 | −194 | −1.29e-4 |
| L4 | combo: sidecar-drop ∘ per-pair freest 50% step3 | 15,284 | 0 | 177,783 | −710 | −4.73e-4 |

The rate gains are MEASURED + CERTAIN (real LZMA re-encode + sidecar truncation). The distortion
cost is the unknown the eval prices. (Note: per-pair coarsening saves FAR fewer bytes than global
because LZMA exploits cross-pair temporal context — coarsening only some pairs keeps the stream
high-entropy. This is itself a finding: the latent rate is a cross-pair-coupled stream, not a
per-pair budget.)

**No-op detector + pre-registered predictions + kill criteria:** see
`submission_dirs_noop_proof.json` + `pre_registered_predictions.json` (recorded BEFORE the eval).

---

## 4. The local advisory kill-gate (48-pair CPU DistortionNet; validated within 9% on the decoder axis)

`local_predicted_band_smoke.py` inflates the frontier + each candidate via the byte-faithful
inflate.py, decodes GT via the contest-exact `frame_utils.yuv420_to_rgb` (R3-corrected), and scores
48 pairs through the EXACT `upstream/modules.py` DistortionNet on CPU. `[macOS-CPU advisory]` —
a SCREEN, not a score claim (the latent distortion changes are small, so the macOS↔Linux FP-drift
caveat is more relevant here than for the large decoder coarsening; the exact contest-CPU eval is
the authority). Frontier 48-pair subset: d_seg 5.384e-4, d_pose 1.190e-5.

| cand | d_seg (Δ) | d_pose (Δ) | dS_dist | dS_rate | **dS_total (advisory)** | verdict |
|---|---|---|---:|---:|---:|---|
| L1 sidecar_drop | 5.689e-4 (+3.05e-5) | 1.319e-5 (+1.30e-6) | +3.63e-3 | −4.04e-4 | **+3.23e-3** | UNFAVORABLE |
| L2 perpair_freest70_step4 | 6.797e-4 (+1.41e-4) | 1.619e-5 (+4.30e-6) | +1.59e-2 | −4.19e-4 | **+1.55e-2** | UNFAVORABLE |
| L3 perdim_freest_step3 | 5.930e-4 (+5.46e-5) | 1.258e-5 (+6.87e-7) | +5.77e-3 | −1.29e-4 | **+5.64e-3** | UNFAVORABLE |
| L4 combo (drop∘perpair50 step3) | 6.089e-4 (+7.05e-5) | 1.432e-5 (+2.42e-6) | +8.10e-3 | −4.73e-4 | **+7.63e-3** | UNFAVORABLE |

**ALL 4 candidates ADVISORY-UNFAVORABLE.** Every advisory dS_total is 690×–3,330× the cross-session
reproducibility floor (4.66e-6) above the favorable threshold — far outside the kill-gate's
validated 9% margin. Ordered least-to-most-bad: **L1 (+3.23e-3) < L3 (+5.64e-3) < L4 (+7.63e-3) <
L2 (+1.55e-2)**.

### The sidecar-rent answer (the canonical finding)
**The 607-byte sidecar PAYS RENT ~8× over.** Dropping it (L1) raises d_seg by +3.05e-5 → +3.63e-3
of score (the `100·d_seg` term), which dwarfs the 607-byte rate gain (−4.04e-4). The 597 tiny
per-pair single-dim corrections (RMS 0.0069 = 0.76% of latent magnitude) are NOT padding — they are
load-bearing fine-tune corrections that recover sub-LSB d_seg precision on a memorized renderer.
This RATIFIES PR101 lesson L27 ("per-pair single-dim correction sidecar... contributes -0.001 to
-0.003 score improvement") on the FRONTIER archive: the sidecar is the cheapest score the frontier
buys per byte, not the freest rate to shave. (No-op detector: L1 inflate proves consumption —
752,614,652 raw frame bytes differ vs frontier; decoder/selector/dqs1 byte-identical.)

### Mechanism (THE LAW decomposition — the deep finding)
The latent axis is genuinely ~20× LESS destructive per byte than the decoder (L1's whole-sidecar
drop costs +3.6e-3 d-dist vs the decoder's gentlest step-2 coarsen +0.069), confirming the latent
is the lower-sensitivity surface the decoder verdict pointed at. BUT it is still net-UNFAVORABLE at
every operating point because (a) the latents feed a MEMORIZED single-video renderer with no
redundant precision, and (b) the latent coding floor is already exhausted (§1), so the only lever
is distortion, and the `100·d_seg` coefficient prices even tiny latent perturbations above the
sub-1KB rate gains the latent axis can offer. The per-pair lever (L2) is the most destructive per
saved byte (aggressive q-coarsen even on free pairs); the per-dim lever (L3) spreads distortion
across all 600 pairs for a small rate gain; the sidecar drop (L1) is the gentlest but still loses
the load-bearing corrections.

---

## 5. EXACT PAIRED CONTEST-CPU EVAL — pinning the sidecar-rent verdict

Per CLAUDE.md MVP-first phasing + the prompt's "ONE paired contest-CPU Modal batch": the local
kill-gate is decisive (all 4 UNFAVORABLE by 690×–3,330× the noise floor), but the canonical
**"does the sidecar pay rent on the EXACT contest scorer?"** question is worth pinning. **L1
(sidecar-drop)** — the gentlest, most canonically-valuable candidate — is dispatched to
`upstream/evaluate.py --device cpu` (Linux-x86_64 Modal CPU container, 1:1 with the contest GHA
runner), single-axis CPU diagnostic (the CPU axis is the leaderboard ranking axis + the authority
for this rate-vs-distortion verdict; L1 is advisory-non-promotable so no CUDA promotion pairing is
warranted). L2/L3/L4 are KILLED LOCALLY by the gate (clearly dominated; the gate saves their spend
— the canonical kill-gate value).

### The decisive measurement (call_id `fc-01KTQTGDF1NZGVMM1R0EKVT036`)

| eval | final_score [contest-CPU] | d_seg | d_pose | bytes | dS_dist | dS_rate | **dS_total** | pays_rent |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| frontier | **0.19198534** | 5.598e-4 | 2.943e-5 | 178,493 | — | — | — | — |
| L1 sidecar_drop | **0.19486984** | 5.893e-4 | 3.060e-5 | 177,886 | +3.288e-3 | −4.042e-4 | **+0.0028845** | **False** |

**L1 exact contest-CPU = 0.19486984 (`score_recomputed_from_components`; 600 samples, returncode=0,
passed=True). dS_total = +0.0028845 ABOVE the frontier — UNFAVORABLE.** The 607-byte sidecar PAYS
RENT: dropping it costs +2.95e-5 d_seg + 1.17e-6 d_pose = +3.29e-3 of score, vs the 607-byte rate
gain (−4.04e-4) — the corrections recover **~7×** their byte cost.

### Rounding-trap caught (apples-to-apples evidence discipline)
The result's `final_score` field is **rounded to 0.19** (a 2-decimal display value) and a naive read
would FALSELY flag L1 as a frontier candidate (0.19 < 0.19199). The authoritative value is
`score_recomputed_from_components = 0.19486984` (matches an independent recompute
`100·d_seg + √(10·d_pose) + 25·B/N` exactly). Per CLAUDE.md "Apples-to-apples evidence discipline":
the rounded display ≠ the precise score; the component recompute is the truth. This averted a
phantom frontier-candidate alert.

### Kill-gate validation
The local 48-pair advisory predicted L1 dS_total = +3.226e-3; the exact 600-pair contest-CPU landed
+2.885e-3 — within **~12%**, and CORRECTLY UNFAVORABLE. This re-validates the local kill-gate (the
decoder verdict measured ~9%; the latent's smaller distortion change widened the gap slightly to
~12%, still well inside the decision margin). The kill-gate correctly killed L2/L3/L4 locally
(advisory +5.6e-3 to +1.55e-2, all decisively dominated) — saving 3 paid Modal dispatches.

---

## 6. Verdict + routing (Catalog #125)

### Verdict: the latent axis is FALSIFIED-AT-IMPLEMENTATION for net rate gain (NOT a paradigm kill — Catalog #307)
The latent distortion lever is intact; the SPECIFIC implementations (sidecar-drop, per-pair coarsen,
per-dim coarsen, combo) are all falsified. L1 scored +0.00289 worse on the contest-CPU authority;
L2/L3/L4 advisory +5.6e-3 to +1.55e-2. THE LAW priced it correctly at every operating point:
distortion > rate.

### The three permanent findings (system intelligence)
1. **The latent coding floor is exhausted AND second-order re-prediction is worse.** No lossless
   lever (LZMA retune / coder swap / 2nd-order delta) recovers material bytes; the authored
   first-order-delta + dict4096-LZMA is optimal. The latent rate is a cross-pair-coupled stream
   (coarsening some pairs barely shrinks it), not a per-pair budget.
2. **The sidecar pays rent ~7×.** PR101 lesson L27 RATIFIED on the frontier: the 607-byte sidecar's
   597 tiny per-pair single-dim corrections (RMS 0.0069) are load-bearing d_seg fine-tunes, the
   cheapest score the frontier buys per byte. Dropping it is the single biggest latent-axis own-goal.
3. **The latent is the lower-sensitivity surface, but still net-unfavorable.** It carries 4.42% of
   distortion sensitivity (NOT 0.02% — the per-byte-uniform |grad| headline ignored pose-weighting),
   ~2× more favorable per byte than the decoder, and ~20× less destructive per byte (L1 whole-sidecar
   drop +3.3e-3 vs decoder gentlest +0.069). But on a MEMORIZED renderer with no redundant latent
   precision and an exhausted coding floor, even tiny latent perturbations are priced above the
   sub-1KB rate gains the axis can offer.

### Routing: DEFER-pending-research (per CLAUDE.md "Forbidden premature KILL")
The latent distortion lever is NOT killed; the naive byte-transform implementations are. Reactivation
criteria (each tests a distinct assumption the frontier doesn't yet exploit):
1. **Score-aware per-element latent QAT** (not byte-transform PTQ): jointly re-train the latents +
   sidecar at a coarser precision to recover the d_seg loss (the canonical low-bit-PTQ-collapse fix).
   Tests "is the coarsened-latent d_seg recoverable by fine-tuning?" — requires the full score-aware
   training loop, not a byte transform.
2. **Latent dimensionality reduction** (fewer than 28 dims, score-aware): if the 28-d latent is
   over-parameterized, a smaller latent could shrink the blob proportionally at controlled d_seg.
   Requires re-training the decoder stem. Tests "is the 28-d latent score-redundant?"
3. **A LARGER sidecar** (the rent direction inverts the attack): the sidecar pays ~7× rent, so
   ADDING more per-pair corrections (more pairs/dims, finer codes) may BUY score below the frontier
   — the opposite of dropping it. Tests "can the sidecar buy more score per byte than it costs?"
   This is the genuinely promising inverted-direction follow-on.

### The frontier remains 0.19198534 [contest-CPU]; no candidate promoted; no submission.

### Wire-in (Catalog #125)
- **Hook #1 sensitivity-map:** NEW per-pair × per-dim latent sensitivity surface
  (`latent_sensitivity_map.json`): the per-pair score-weighted sensitivity (2,850× range; freest
  pairs 219/177/260/...) + per-dim frame-MSE Jacobian (2.2× range; freest dims 20/14/26/...). Permanent
  correction: the latent carries 4.42% of distortion sensitivity, and the prompt's protected-pair list
  (510–522/133/177–178) was ledger-FALSIFIED (177 is the freest pair; 512/514/522 are low).
- **Hook #3 bit-allocator:** the per-pair-allocation prior is confirmed NECESSARY-but-INSUFFICIENT
  (same as decoder): per-pair sensitivity ranking is correct, but the cross-pair-coupled LZMA stream
  + memorized-renderer d_seg sensitivity dominate any sub-1KB rate gain. The allocator must use a
  score-aware fine-tune recovery model, not a byte-transform PTQ.
- **Hook #5 continual-learning:** 1 exact contest-CPU `tac.action_effect.v1` row minted via
  `ingest_exact_eval_to_candidate.py` (`pays_rent=False, above_frontier, binding=rate`) — the FIRST
  latent-axis PR110++ exact row; reseeds the V3 ΔS-judge that latent byte-transform PTQ is dominated.
- **Hook #6 probe-disambiguator:** RESOLVED "is there free rate in the latents?" → NO via byte
  transform; the binding constraint is the exhausted coding floor + memorized-renderer d_seg
  sensitivity. Also resolved (advisory) that the local macOS-CPU kill-gate prices latent distortion
  within ~12% (validating the kill-gate at smaller distortion scale).
- **Hook #2 Pareto:** the frontier is on its latent-distortion Pareto edge — d_seg cannot be traded
  for rate via byte-transform latent coarsening; the latent is rate-AND-distortion saturated.
- **Hook #4 cathedral-autopilot:** N/A — no promotable archive (all candidates above frontier).

### Cost
Per-pair/per-dim sensitivity map + RD probes + local 48-pair kill-gate = $0 (macOS-CPU). 1 Modal CPU
eval (L1 only; L2/L3/L4 killed locally by the gate) ≈ $0.3, under the $5 STOP gate. The kill-gate
saved 3 paid dispatches (~$0.9). First L1 dispatch failed-fast at the paired-by-default guard (rc=1,
no spend); re-dispatched with `--single-axis-waiver-reason` (CPU diagnostic, non-promotable).


