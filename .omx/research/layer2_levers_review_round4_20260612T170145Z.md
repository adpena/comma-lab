# Recursive adversarial review — ROUND 4 of the 5 Layer-2 levers (2026-06-12)

**Reviewer:** R4 subagent (author ≠ reviewer). Prior rounds, all on disjoint lenses:
R1 (`layer2_levers_independent_audit_20260612T151829Z.md`, `4cbd9676a`) = static NO-FAKE (all 5 REAL,
no HIGH, daemon-safe, 97 tests). R2 (`...154002Z.md`, `253f8ab9a`) = runtime/resume (fixed a
Lever-4-EMA-not-persisted MEDIUM, `6e0d8feff`). Gap-closure (`990fd3de3`) = MED-1 scan-order
(Spearman −0.14→0.90) + MED-2 byte-direction + LOW-1 doc. R3 (`...164500Z.md`, `d5cadcb31`) =
gradient-direction (all levers descend their real target or hold flat-at-optimum; double-counting
COHERENT; fixed a LOW compose-test-timeout-flake marker).
**R4 has a FOURTH, distinct lens: the DEPLOYED ARCHIVE end-to-end (not just the loss) + multi-stage-
boundary LIVE behavior + a fresh-eyes holistic re-read.** The bug class R4 hunts: a train/deploy gap at
the ARTIFACT level — the archive an all-5-on run produces decodes to something DIFFERENT from what
training picked as BEST, or a lever activation at a stage boundary destabilizes the run. Unreachable by
a loss/gradient/resume audit.
**Scope:** VERIFY + TEST; new $0 probes only (no lever-file edit was needed — zero findings). Did NOT
touch `src/tac/substrates/cool_chic/**` (Track B), the basin daemon (pid 33911, confirmed ALIVE 5h25m+,
default config `--no-split-by-head --train-device mps`, untouched), or its out-dir.
**Authority:** every in-loop / synthetic number here is `[macOS-CPU advisory]` NON-PROMOTABLE (synthetic
scorer, RESEARCH-ONLY); the levers land MEANS, the exact frontier is UNMOVED (`0.19109982`). Mission
contribution: `frontier_protecting` (proves the deployed all-5-on artifact is faithful + boundary-stable
for the multi-day run).

## CLEAN-PASS VERDICT: **CLEAN → counter ADVANCES to 1/3.**

R4 found **ZERO findings** (no HIGH, no MEDIUM, no LOW). All four lenses passed. R4 starts from R3's fixed
code (the compose-timeout marker `d5cadcb31`), exercises the deployed-archive + boundary + fresh-eyes
surfaces the prior 3 rounds did NOT, and finds nothing to fix. Per the protocol ("3 consecutive clean
passes required"), this is the **first clean pass** — counter 0/3 → **1/3**. The two new probes are durable
evidence artifacts (ruff-clean); no lever source changed, so byte-identity is structurally untouched.

---

## A. DEPLOYED-ARCHIVE END-TO-END under ALL-5-ON (the headline R4 lens) — PASS.

Probe `experiments/probe_r4_deployed_archive_all_five.py`: a REAL all-5-on driver run (FiLM + seg
surrogate + T-anneal 1.0→0.05 + rate w+lat + score-aware QAT + margin τ=2.0 + C1a, synthetic scorer,
2 epochs, n_pairs=8) PRODUCES an archive, then I take THAT archive's bytes and prove 7 deploy properties:

| # | Deployed-archive property | Result |
|---|---------------------------|--------|
| 1 | all-5-on run produces a best archive | 21398 B produced |
| 2 | `parse_archive` succeeds; FiLM weights IN the decoder blob | 4 `pose_film.*` keys present |
| 3 | **pose-FiLM pose section WRITTEN + parseable (not dropped)** | shape (8,6) present |
| 4 | numpy-portable `inflate_film_decoder` decodes to valid frames | (8,2,3,384,512) ∈ [87.9,165.6], all finite |
| 5 | the deployed archive is SCOREABLE (finite score components) | score=81.79 / d_seg=0.80 / d_pose=0.30 / rate=5.7e-4 |
| 6 | **eval(cursor)==inflate render parity (the train/deploy seam)** | `allclose` atol 1e-4 — NO skew |
| 7 | tail-batch (5+3) cursor render == inflate render | aligned (pose lookup correct at the tail) |

**The headline verdict: all-5-on produces a VALID, SCOREABLE archive whose decode MATCHES training.**
The load-bearing interpretations (each verified, not assumed):

1. **The pose-FiLM section is written AND consumed at inflate (property 3+4+6).** The produced archive
   carries the additive pose section (`parse_pose_section` returns shape (8,6), not None) AND the FiLM
   weights ride in the decoder blob (4 `pose_film.*` keys); `inflate_film_decoder` reads BOTH back,
   rebuilds the FiLM wrapper, sets the stored pose, and renders FiLM-conditioned frames. The pose is NOT
   silently dropped between training and deploy.
2. **The BEST-selection eval render is BIT/atol identical to the deployed inflate render (property 6 — the
   crux).** Training picks BEST via the scorer's `exact_eval`, which routes the FiLM decoder through a
   cursor-based `_FiLMEvalDecoder` (cursor reconstructs per-pair index per batch). The contest scores the
   archive via `inflate_film_decoder`. I rebuilt the eval decoder FROM the produced archive bytes (so it is
   provably the DEPLOYED artifact) and rendered it with the EXACT vendored call pattern (`.eval()` resets
   the cursor, then `decoder(z)` per batch in strict pair order) — the cursor render `allclose` the inflate
   render at atol 1e-4. **No eval/deploy skew: training does not pick a BEST the contest fails to score.**
3. **The cursor contract holds on the REAL vendored `evaluate_decoder` (static trace, the load-bearing
   seam).** I read the vendored `score.evaluate_decoder` (the REAL contest eval path,
   `…/public_pr95_intake_…/src/score.py`): it calls `decoder.eval()` EXACTLY ONCE at entry (→ resets the
   `_FiLMEvalDecoder` cursor to 0) and `decoded = decoder(z)` EXACTLY ONCE per batch, iterating
   `pair_idx` in strict order `0, B, 2B, …` with `z = latents[arange(pair_idx, pair_idx+B)]`. The cursor
   reconstructs the SAME `arange(cursor, cursor+B)` and advances by `z.shape[0] == B`. Even at the tail
   (a short final GT batch), `B = batch_gt.shape[0]` equals the latent batch size, so alignment holds — I
   verified this with the split-batch (5+3) probe (property 7). The eval/inflate parity is therefore not an
   accident of the synthetic batch size; it is structural on the REAL path.
4. **QAT score-aware grid decode-match is SOUND (no train/deploy weight mismatch).** Fresh read of
   `driver._train_one_epoch:556-587`: score-aware QAT applies `apply_score_aware_qat` (STE fake-quant of
   `weight.data`) for the FORWARD ONLY, then `restore_score_aware_qat` restores the cloned float weights
   BEFORE `optimizer.step()`. The archived EMA shadow tracks the FLOAT weights; the codec ALWAYS
   127-requantizes that same float shadow (`quantize_state_dict`, N=127). So there is NO separate "QAT grid"
   in the deployed archive to mismatch — the eval and inflate decode the SAME 127-requantized EMA float
   weights (property 6 confirms bit-identity on the produced archive, which went THROUGH the 127-requant).
   `restore` is a real clone-and-restore (`originals[name]=mod.weight.data.clone()` → `mod.weight.data=
   originals[name]`), so the optimizer never steps on fake-quant'd weights. The Lever-4 byte win remains the
   honestly-disclosed indirect effect (MED-2 caveat in the docstring) — but there is no ARTIFACT-level
   decode divergence.

## B. MULTI-STAGE-BOUNDARY LIVE behavior — PASS.

Probe `experiments/probe_r4_multistage_boundary.py`: a 2-stage curriculum (stage 1 = ALL LEVERS OFF
vendored, 4 ep; stage 2 = ALL 5 LEVERS ON, 6 ep, carrying decoder/latents/EMA across the boundary).
R2 checked phase interactions STRUCTURALLY; R4-B checks the ACTUAL archive/loss behavior crossing the
boundary where the levers turn ON.

| # | Boundary property | Result |
|---|-------------------|--------|
| A | Lever-2 anneal RESTARTS per stage (epoch_in_stage resets) | stage2 T 1.000→0.050; stage1 static — no carry-over |
| B | all epochs finite across the boundary | 10/10 finite; stage1 loss 163.22→163.19, stage2 78.91→78.92 |
| C | no DIVERGENCE SPIKE when levers activate | stage2/stage1 loss ratio **0.48×** (well under the 50× reject bar) |
| D | end-of-run (levers-on stage) archive valid + inflatable | 21388 B, frames (8,2,3,384,512) finite, pose present |

**The boundary verdict: lever activation at the stage boundary produces a COHERENT CONTINUING descent.**
The loss actually DROPS at the boundary (163→79, ratio 0.48×) — the soft-cosine seg surrogate + margin
weight is a smaller-magnitude objective than the stage-1 CE, so the transition is smooth, NOT a
discontinuity/divergence. The anneal correctly restarts at T=1.0 in stage 2 (driven by `epoch_in_stage`,
which resets to 0 for a new stage at `driver.py:1195` `start_epoch=…0`, passed as `epoch_in_stage=epoch`
at `:1251`) — it does not continue cold from stage 1's end. The decoder/latents/EMA carry across
(`carry_decoder`/`carry_latents`, `:1230-1232`); the QAT sensitivity EMA is correctly rebuilt fresh per
stage (`_build_stage_runtime`) and re-seeded within the QAT stage — the R2 fix carries it across a
death/resume, but a fresh STAGE legitimately starts a fresh EMA (QAT only runs on QAT stages).

## C. FRESH-EYES HOLISTIC RE-READ (question all interpretations) — PASS (1 un-covered path closed).

A senior-engineer full re-read challenging what the prior rounds took for granted:

1. **FiLM × optimizer-param-set × Muon (a path R1-R3 did NOT exercise — they all used `use_muon=False`).**
   The REAL levers-active stages (PR95 stages 5-8) use Muon. Two fresh checks:
   - **Muon partition covers ALL FiLM params (0 dropped).** `partition_params_for_muon` (vendored) routes
     2D non-stem/non-rgb weights to Muon, biases/1D to AdamW. On a FiLM-wrapped decoder: 30 trainable
     params, 12 Muon + 18 AdamW = 30 covered, **0 uncovered**. The 4 `pose_film.*` params are all covered
     (fc1/fc2 weights → Muon, biases → AdamW). FiLM is genuinely optimized under Muon stages — no silent
     freeze of the FiLM MLP.
   - **all-5-on + Muon + FiLM (zero-init fc2 through Newton-Schulz) is NO-NaN.** A 3-epoch all-5-on run
     WITH `use_muon=True` (the configuration R2's 80-epoch compose run did NOT cover — it was AdamW-only)
     completes: best_score 81.76, 21385 B archive, inflate frames finite, pose section present. The zero-init
     FiLM fc2 fed through Muon's Newton-Schulz orthogonalization does not produce NaN. This closes the
     last un-exercised lever-interaction path.
2. **`--levers all --self-test` on current HEAD — PASS.** Re-confirmed R2's lens-D on the gap-closure +
   R3 HEAD: `lever1_rate_surrogate=true, lever2_seg_surrogate=soft_cosine, lever2_temperature_anneal=true,
   lever3_pose_film=true, lever4_score_aware_qat=true, lever5_margin_weight=true`, archive exists, pose
   section parses, self_test=PASS. No lever silently dead on the real launch path. (The 1.8 M synthetic
   self-test scratch dir was cleaned up — rebuildable via the same command.)
3. **`codec_scan_order=True` is hardwired in the driver (the MED-1 deploy-faithful fix is always on).**
   `driver.py:713` `rate_cfg = RateSurrogateConfig(codec_scan_order=True)` — whenever the rate lever fires,
   it uses the deploy-faithful scan order (Spearman 0.90 vs real bytes), not the legacy −0.14 per-tensor
   mode. No path reverts to the slack proxy.
4. **StageSpec lever defaults are all OFF (`_spec_from_stage_config` sets none of them) — re-confirmed.**
   The vendored curriculum / basin daemon is fully inert; only `--levers all`/explicit StageSpec fields turn
   levers on. The default byte-identity proof (`test_default_train_epoch_matches_vendored_only_reference`)
   holds in the full suite (§D).

## D. R1/R2/R3 INVARIANTS RE-CONFIRMED ON CURRENT HEAD — HOLD.

Full suite (lens D):
```
.venv/bin/python -m pytest src/tac/torch_vehicle/tests/ src/tac/tests/test_rate_surrogate.py -q --timeout=400
→ 104 passed in 136.45s
```
**0 failures, 0 flakes** (the R3 compose-timeout marker holds under normal load). This subsumes: the
all-default byte-identity proof, the FiLM-off byte-identical archive, R2's Lever-4-EMA-resume round-trip,
the eval==inflate parity test, and all 13 pose_film wire-in tests + the lever behavior tests.

## Findings by severity

- **HIGH:** NONE. No deployed-archive train/deploy divergence, no pose-section-dropped, no QAT-decode-
  mismatch, no eval/inflate skew, no boundary divergence, no regression.
- **MEDIUM:** NONE.
- **LOW:** NONE.

## The deployed-archive verdict (the R4-A deliverable)

**VALID + SCOREABLE + FAITHFUL.** All-5-on produces a real archive that (a) parses back, (b) inflates to
finite scoreable frames, (c) carries the FiLM pose section written-and-consumed, and (d) whose
BEST-selection eval render is bit-identical (atol 1e-4) to the deployed `inflate_film_decoder` render —
including the tail-batch case — on the REAL vendored `evaluate_decoder` cursor contract (statically
traced). The QAT score-aware grid does not introduce an artifact-level decode mismatch (the codec always
127-requantizes the same float EMA shadow). **No train/deploy gap at the artifact level.**

## The multi-stage-boundary verdict (the R4-B deliverable)

**COHERENT + STABLE.** Lever activation at a stage boundary produces a continuing descent (loss ratio
0.48×, no spike), the anneal restarts per-stage (T 1.0→0.05 in the new stage), decoder/latents/EMA carry
across, and the levers-on stage produces a valid inflatable archive. No boundary discontinuity/divergence.

## Test-run count

- Full suite (lens D): **104 passed in 136.45s, 0 failures.**
- R4-A deployed-archive probe: **7/7 checks PASS** (incl. eval==inflate parity + tail-batch).
- R4-B multi-stage-boundary probe: **4/4 checks PASS.**
- Fresh-eyes (lens C): all-5-on+Muon+FiLM no-NaN run COMPLETE; Muon partition 0-dropped; `--levers all
  --self-test` PASS; 3 deploy tests (eval==inflate parity + inflate round-trip + FiLM-off byte-identity)
  PASS in isolation.

## Probes added this round (durable evidence artifacts; no lever source touched)

- `experiments/probe_r4_deployed_archive_all_five.py` — the 7-check deployed-archive end-to-end probe.
- `experiments/probe_r4_multistage_boundary.py` — the 4-check multi-stage-boundary live probe.

Both ruff-clean. No edit to any lever file → byte-identity of the default/daemon path is structurally
untouched (no re-run needed; nothing changed). The 1.8 M synthetic self-test scratch from the
`--self-test` invocation was cleaned up (rebuildable).

## Wire-in / provenance

6-hook (Catalog #125): all N/A — this is a review-round memo + two $0 evidence probes (no new score-claim
surface; the levers' own hooks are in the landing memo). Mission contribution: `frontier_protecting`
(verifies the deployed all-5-on artifact is faithful + boundary-stable for the multi-day descent; the END
remains a lower exact score, frontier UNMOVED `0.19109982`). Authority: all numbers `[macOS-CPU advisory]`
synthetic-scorer NON-PROMOTABLE. No GPU launched, no daemon touched (pid 33911 ALIVE 5h25m+ + untouched),
no Cool-Chic touched.

**VERDICT: CLEAN (zero findings) → counter ADVANCES 0/3 → 1/3.** R5 is the next chance to reach 2/3.
The deployed all-5-on archive is valid, scoreable, and faithful (eval==inflate); the multi-stage boundary
is coherent and stable. R5 should pick a FIFTH distinct lens (e.g. real-scorer paired smoke on a tiny
real-video slice; numerical-precision/device-drift of the FiLM render MPS-vs-CPU; or an adversarial
re-read of the rate-surrogate codec-scan-order math under extreme weight distributions) to keep the
clean-pass count meaningfully adversarial rather than re-running covered ground.
