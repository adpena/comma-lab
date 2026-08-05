---
arm: ddm_ar12 (MAIN-AUTHORED — codex fleet down until 2026-08-10; complete-read directive)
source: "rezabyt.github.io/blogposts/sigreg-tutorial.html — 'SIGReg from First Principles',
  Reza Bayat, 2026-07-21"
underlying: "SIGReg from LeJEPA (Balestriero & LeCun 2025); applied in LeWorldModel
  (Maes et al. 2026)"
utc: 2026-08-05
read_depth: FULL-TEXT (2 WebFetch passes: reduction chain + implementation/training loop)
research_only: true
score_claim: false
promotion_eligible: false
pointer_moved: false
axis: "[paper-crosswalk scorer-free]"
tokens: "[no-triality] [p0-ledger-ok]"
---

# AR12 Receipt — SIGReg tutorial (LeJEPA anti-collapse), MAIN deep read

## Answer First

SIGReg is a differentiable scalar regularizer whose population minimum is the isotropic
Gaussian N(0, I_D): per step, sample M random unit directions (fresh each step), project
embeddings to 1D, and score each projection with the **Epps–Pulley characteristic-function
statistic** `T = N Σ_k α_k w(t_k) [(C_k − e^{−t_k²/2})² + S_k²]` (empirical CF vs target
CF, trapezoid quadrature K knots on t ∈ [0.2, 4.0], weight w(t)=e^{−t²/2λ²}, λ=1). The
Cramér–Wold theorem makes all-1D-projections equivalent to the full D-dim distribution;
cost O(MKN); fully autograd-differentiable; composes as `L_pred + 0.1·SIGReg` with no
stop-gradient. Anti-collapse argument: N(0,I_D) is full-rank, collapse requires zero
eigenvalues, so collapse is structurally incompatible with the regularizer's global
minimum.

**The one-sentence verdict for us:** the GAUSSIAN TARGET is anti-adopted (it is the
maximum-entropy distribution — directly opposed to a rate-minimized counted payload),
but the EP **machinery is target-agnostic** (any φ₀ works), which yields one genuine
candidate: a differentiable "stay-compressible" distributional regularizer with φ₀ set
to the CODER-optimal token distribution — plus a cheap distributional instrument, and a
named cure-class for any future latent-surrogate training that collapses.

## Ranked Crosswalk

| rank | disposition | claim → Pact surface | named consumer | falsifier | cost |
|---:|---|---|---|---|---|
| 1 | **ANTI-ADOPT, write it down** | "SIGReg on the TR1 token field" is a trap: its target N(0,I) is MAXIMUM-entropy; our counted tokens want MINIMUM coder-cost. Naive adoption fights the rate term head-on. Recorded so no future arm name-borrows it (the m52 binary-judgment sister: the technique is not "good/bad", its TARGET is vehicle-opposed). | any future token-regularizer proposal | n/a (guard row) | $0 |
| 2 | **ADOPT-AS-CANDIDATE (novel adaptation, race-gated)** | The EP statistic accepts ANY target CF φ₀. Replace e^{−t²/2} with the CF of the RATE-OPTIMAL token distribution (fit from the shipped coder's own statistics, e.g. the Brotli/SMEVR-winning empirical token histogram) → a differentiable distribution-matching rate surrogate: pushes trained tokens toward the distribution the live coder prices cheapest. This is a RIVAL to the QA86/b2b SMEVR rate-in-loss surrogate and enters ONLY through a same-window race against it (races-not-reputation, #940 doctrine). Bonus: its collapse-repelling structure also targets the dead-codeword pathology (#873/dc1 — dead codewords, not mode-share, were the measured rate defect) from the training side. | burn arm-matrix rate-in-loss slot (raced vs SMEVR surrogate) · #873 dead-codeword vein | Race falsifier: at matched window/bytes, EP-target surrogate produces ≥ real coded bytes vs the SMEVR surrogate arm; or its gradients destabilize seg-hold (lg1 fires). Also dies if the coder-optimal φ₀ is so concentrated that EP gradients vanish (λ-bandwidth check first, $0). | small build (EP kernel is ~20 lines, O(MKN)) + one raced window |
| 3 | **ADOPT-AS-INSTRUMENT ($0)** | The EP statistic as a cheap calibrated distributional-drift alarm and diagnostic: differentiable, target-agnostic, O(NK) per direction. Complements AR11-P1 (spectral instrument tests SPATIAL frequency content; EP tests VALUE distributions). Sister of cf1's (#952) conformal L1 alarm-calibration adoption — EP gives a second, characteristic-function-based test with an explicit bandwidth knob (λ=0.3 ≈ moments-only, λ=1 catches shape). | telemetry/confound alarms (L1 layer) · render-vs-solve value-distribution diff for dw1 | Instrument shows nothing the existing conformal alarms don't, on the first real telemetry replay → folded. | $0 |
| 4 | **CURE-CLASS ON FILE (conditional)** | If any latent-surrogate/distill training we run (dw1 distill window, #485 JEPA-latent costate surrogate, any future ξ-world-model per the m02 robotics-worldmodel archetype — LeWM is the direct reference) exhibits representational collapse, SIGReg with the ISOTROPIC target is the published cure, and it joins the vae1 (#769) posterior-collapse↔inert-cells vein as the JEPA-side member. Here the Gaussian target is CORRECT because a surrogate's latents are not counted bytes. | dw1 · #485 · vae1 vein | n/a until a collapse is observed (conditional row) | $0 now |
| 5 | **FOLDED (method internals)** | Cramér–Wold sketching (M random 1D projections, error ~1/√M, M=16 already usable), quadrature details (K knots on [0.2,4.0], endpoints Δt/2), fresh-directions-per-step, finite-batch floor O(1/N), bandwidth sensitivity (λ=0.3 passes bimodal if variance matches — a WARNING for rank-3 use: use λ≥1), single-direction 3× variance (always average the sphere). Retained as implementation facts for rank-2/3 builds. | rank-2/3 builders | — | — |

## Constants-are-poison note

Their λ=1.0, M=1024, K=16, t∈[0.2,4.0], λ_reg=0.1 are THEIR-vehicle constants. Any
rank-2/3 build derives: t-range from the target φ₀'s decay (their [0.2,4.0] rationale is
w(4)≈3e-4 — re-derive for our λ and φ₀), M from the tolerated per-step direction noise,
λ_reg from the measured gradient-share protocol (#312 stage-boundary weights), never
copied.

## RECALL EVIDENCE

- Consumers recalled before ranking: QA86/b2b SMEVR rate surrogate (the rank-2 rival) ·
  #873/dc1 dead-codeword defect + ms8 degeneracy correction · #940 races-not-reputation ·
  #485 JEPA-latent surrogate probe · vae1 posterior-collapse↔inert-cells vein (#769) ·
  cf1 conformal alarm adoption (#952) · dw1 distill line · lg1 seg-hold guard · m02
  robotics-worldmodel archetype · m52 never-binary-judgment · AR11 receipt (the spectral
  instrument rank-2 complements) · #312 gradient-share weighting protocol.
- Scoped negative: no prior SIGReg/LeJEPA/Epps-Pulley receipt found in `.omx/research`
  or memory (queries: `SIGReg`, `LeJEPA`, `Epps`, `characteristic function`,
  `Cramér-Wold`, `anti-collapse`). The vae1 receipt covers VAE-side collapse only.

## Boundaries

Full tutorial read via 2 targeted passes; underlying LeJEPA/LeWM papers NOT separately
fetched (the tutorial reproduces the method's equations and constants; a full LeJEPA read
fires only if rank-2 survives its $0 bandwidth check). No code run, no scorer forward,
no launch, no lane claim.

## NEXT_IF_RESUMED

1. **AR12-P0 ($0, gates rank-2):** fit the empirical CF of the CURRENT sub_final token
   stream; check EP gradient magnitude at that φ₀ across λ ∈ {0.5, 1, 2} — if vanishing,
   rank-2 dies before any build.
2. **AR12-P1 (build, race-gated):** EP-target rate surrogate as a DSL lever, raced vs the
   SMEVR surrogate at the next burn window per rank-2's falsifier.
3. **AR12-P2 ($0, opportunistic):** EP value-distribution diff of renders vs C1 solve
   frames alongside AR11-P1's spectral diff (one script, two instruments, same cached
   frames).

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
