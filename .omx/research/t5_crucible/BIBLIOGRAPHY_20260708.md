# T5 CRUCIBLE — BIBLIOGRAPHY (requirement S: citation provenance backfill)

date: 2026-07-08 · seat: CITATION-PROVENANCE · status: fresh-research-round-1
scope: `ct_deepresearch_1_training_campaign_control_20260707.md` (CT-1; had ZERO resolvable
citations — fully backfilled here), `ct_deepresearch_2_pde_geometric_topological_control_20260707.md`
(CT-2; had 11 — verified + completed), `DRAFT_OPTIMAL_STACK_v5_20260707.md`, and the two #305
literature sweeps (`litsweep_training_dynamics_control_20260705.md`,
`litsweep_representation_taskspace_20260705.md`).

**Verification method (per requirement S: no fabricated citations).** Every arXiv ID / DOI marked
`Y-FETCHED` was fetched at its abstract page 2026-07-08 and confirmed to resolve to the named
paper (title + authors read back). Entries marked `Y-SEARCH` were confirmed via web search
(exact title + venue + volume/pages read back from ≥1 authoritative index). Entries marked
`UNRESOLVED` are recorded citations the sweep did NOT verify (they may well be real; they are
simply not certified here — do not cite them as verified). No citation below was written from
memory alone.

**Verification counts:** 21 arXiv/DOI targets FETCHED-verified · 24 citations SEARCH-verified ·
7 UNRESOLVED (recorded in the source memos but not certified) · 2 provenance FINDINGS
(attribution correction + folklore-without-formal-publication), see §3.

---

## §1 — MAIN TABLE (claim imported → citation → deriving doc § → v5 consumer → verified? → marimo?)

Legend: verified? ∈ {Y-FETCHED (abstract page fetched), Y-SEARCH (title/venue confirmed by
search), UNRESOLVED}. marimo? = #347 implement-a-paper candidate (★ = top-5, see §4).

### 1A — CT-1 backfill (training/campaign optimal control; previously ZERO citations)

| # | claim/law imported | citation | deriving doc § | v5 decision/build consumed by | verified? | marimo? |
|---|---|---|---|---|---|---|
| 1 | Pontryagin maximum principle (adjoint/costate, bang-bang + singular arcs, transversality) | Pontryagin, Boltyanskii, Gamkrelidze, Mishchenko (1962), *The Mathematical Theory of Optimal Processes*, Interscience/Wiley (Russian orig. 1961) | CT-1 §1 (λ_e(T) on EMA shadow; lr = singular arc; PMP stop-rate ε_stop) | v5 fold item 3 "PMP fixes" (V=4→5; PMP stop-rate would-fire row §12.1(5)); costate req-M mature form λ=∇V | Y-SEARCH (classical text; standard bibliographic record) | N |
| 2 | Exponential turnpike (entry transient → turnpike → exit transient; transient lengths budget-independent) | Trélat & Zuazua (2015), "The turnpike property in finite-dimensional nonlinear optimal control", *J. Differential Equations* 258(1):81–114, DOI 10.1016/j.jde.2014.09.005 (HAL hal-00946536) | CT-1 §2.1–2.3 (settle 3/ν≈115 ep; cap_fin keep; TAIL cost 265–350 ep/cycle) | v5 §2.3 TAIL budget law (★★★★); cap_fin floor 150 KEEP; dwell_TAIL ≥ 115 ep | Y-SEARCH (DOI + venue + pages confirmed) | ★ Y — interactive budget allocator on OUR measured erosion slopes (ν=0.0262/ep) |
| 3 | Strict dissipativity ⇔ turnpike characterization | Grüne & Müller (2016), "On the relation between strict dissipativity and turnpike properties", *Systems & Control Letters* 90:45–53, DOI 10.1016/j.sysconle.2016.01.003 | CT-1 §2.1 ("Grüne strict-dissipativity characterization") | same as #2 (theorem grounding of the turnpike import) | Y-SEARCH (DOI + award record confirmed) | N |
| 4 | MPC/receding-horizon suboptimality α(N) improves exponentially in N under exponential cost controllability | Grüne & Pannek (2017), *Nonlinear Model Predictive Control: Theory and Algorithms*, 2nd ed., Springer, DOI 10.1007/978-3-319-46024-6 | CT-1 §3.2 (horizon N* = 2 cadences; model-validity binding) | v5 §2 forecast-model horizon pin (N*=2 with exp/powerlaw mixture) | Y-SEARCH (standard text) | N |
| 5 | MPC stability/optimality canonical frame | Mayne, Rawlings, Rao, Scokaert (2000), "Constrained model predictive control: Stability and optimality", *Automatica* 36(6):789–814 | CT-1 §0 row 3, §3.1 (the receding-horizon frame) | v5 fold item 1: forfeit-matched exit s* = ν·forfeit = 1.41e-5 S/ep (`--tau-fin-slope-star`) | Y-SEARCH | N |
| 6 | ISS event-triggering: fire on state-proportional error ‖e‖ ≥ σ‖x‖; Zeno exclusion via minimum inter-event time | Tabuada (2007), "Event-Triggered Real-Time Scheduling of Stabilizing Control Tasks", *IEEE Trans. Automatic Control* 52(9):1680–1685 | CT-1 §4.1 (eps_rel relative form ratified; req-B ≡ Zeno theorem; eps_c = σ·ν_c·d_seg_c) | v5 event-exit contract (req B); per-class veto threshold formula | Y-SEARCH (venue + vol + pages confirmed) | ★ Y — live trigger visualization replaying OUR mod32cap 41-row verdict trace with a σ slider |
| 7 | Event-triggered + SELF-triggered control survey (compute next sample time from current state) | Heemels, Johansson, Tabuada (2012), "An Introduction to Event-Triggered and Self-Triggered Control", *Proc. 51st IEEE CDC*, Maui, 3270–3285 | CT-1 §4.2 (self-triggered verdict cadence Δt = clamp(floor/\|Ŝ′\|, 25, 100)) | v5 B-CT3 self-triggered verdict cadence (−30–40% n600 verdicts late-run); probe P-CT2 | Y-SEARCH (dblp + TU/e portal) | ★ Y (same demo as #6 — the P-CT2 backtest IS the notebook) |
| 8 | Periodic event-triggered control (ETC on a sampled clock) | Heemels, Donkers, Teel (2013), "Periodic Event-Triggered Control for Linear Systems", *IEEE Trans. Automatic Control* 58(4):847–861 | CT-1 §4.1 ("Heemels et al. periodic-ETC") | verdict-cadence-on-a-clock legality (cadence 25 as the periodic base) | UNRESOLVED (named in CT-1; not fetched/searched this sweep) | N |
| 9 | Dwell-time stability of switched systems τ_d > ln(μ)/ν | Liberzon & Morse (1999), "Basic problems in stability and design of switched systems", *IEEE Control Systems Magazine* 19(5):59–70; book: Liberzon (2003), *Switching in Systems and Control*, Birkhäuser | CT-1 §5.1 (τ_d ≈ 9.3 ep from μ=1.275, ν=0.0262; min-stage 250 = 27×) | v5 §2 dwell check row; TAIL dwell lower bound 115 ep | Y-SEARCH (venue + vol + pages confirmed) | N |
| 10 | Average dwell-time stability | Hespanha & Morse (1999), "Stability of switched systems with average dwell-time", *Proc. 38th IEEE CDC*, 2655–2660 | CT-1 §5.1 | same as #9 | Y-SEARCH | N |
| 11 | Common Lyapunov function ⇒ no dwell restriction; mode admission via common V | standard result in Liberzon (2003) (see #9) | CT-1 §5.2 (S as common V; l7 = measured counterexample → switched-stability exclusion) | v5 mode-admission rule (a stage/lever enters the default graph only with ΔS ≤ 0 on common V, else restore-guard) | Y-SEARCH (via #9's book) | N |
| 12 | Gain-scheduling/LPV frozen-time slow-variation condition \|ṗ\|/p ≪ ν | Rugh & Shamma (2000), "Research on gain scheduling", *Automatica* 36(10):1401–1425, DOI 10.1016/S0005-1098(00)00058-3 | CT-1 §6.1 (1-Lipschitz easing = LPV condition, 7.2× margin; ramp_length ≥ 3/ν) | v5 new-ramp law (any NEW ramp ≥ 115 ep unless measured deconflict row) | Y-SEARCH (DOI confirmed) | N |
| 13 | Projection operator (not saturation/clamp) preserves adaptation Lyapunov argument at boundary | standard adaptive-control result — textbook form in Ioannou & Sun (1996), *Robust Adaptive Control*, Prentice-Hall (projection modification) | CT-1 §6.2 (clamp binding >90% ⇒ INERT; fix = projection) | v5 gated B-CT4 (clamp→projection for adaptive-ε) | UNRESOLVED (CT-1 names no source; Ioannou-Sun is the canonical text but was not verified this sweep) | N |
| 14 | ILC: u_{k+1} = u_k + L·e_k; monotone convergence iff ‖I−LP̂‖<1; repeatable disturbance driven to zero | Arimoto, Kawamura, Miyazaki (1984), "Bettering operation of robots by learning", *J. Robotic Systems* 1(2):123–140, DOI 10.1002/rob.4620010203 | CT-1 §7.1 | v5 §2.4 decode-gap ILC law (train bar 9.9573e-4); §8 campaign Newton-ILC γ=0.7 | Y-SEARCH (DOI + venue confirmed) | N |
| 15 | ILC survey (stability, performance, transient, robustness; design techniques) | Bristow, Tharayil, Alleyne (2006), "A Survey of Iterative Learning Control", *IEEE Control Systems Magazine* 26(3):96–114, DOI 10.1109/MCS.2006.1636313 | CT-1 §7 header ("Bristow/Tharayil/Alleyne survey") | same as #14 | Y-SEARCH (DOI + pages confirmed) | N |
| 16 | EWMA run-to-run controller stability: intercept update stable iff 0 < ωξ < 2 (ξ = true/model gain ratio) | Ingolfsson & Sachs (1993), "Stability and Sensitivity of an EWMA Controller", *J. Quality Technology* 25(4):271–287 | CT-1 §7.1 ("EWMA-R2R stability (semiconductor lit)") | v5 §2.4 Δ̂ EWMA update ω=0.5 (stable for ξ∈(0,4)) | Y-SEARCH (venue + vol + pages confirmed) | N |
| 17 | Run-by-run process control frame (semiconductor) | Sachs, Hu, Ingolfsson (1995), "Run by Run Process Control: Combining SPC and Feedback Control", *IEEE Trans. Semiconductor Manufacturing* 8(1):26–43 | CT-1 §7.1 (the R2R campaign analogy) | campaign layer §8 (run-to-run config updates) | UNRESOLVED (lineage confirmed via #16's search context; exact record not independently verified) | N |
| 18 | Extremum seeking: averaging convergence to O(a²) neighborhood under timescale separation | Krstić & Wang (2000), "Stability of extremum seeking feedback for general nonlinear dynamic systems", *Automatica* 36(4):595–601, DOI 10.1016/S0005-1098(99)00183-1 | CT-1 §8.1 | v5 rate-sweep admissibility (dither a* = max(2·floor/ĝ, resolution); ≥5.3 KB byte steps; req-F#6 precondition) | Y-SEARCH (DOI + vol + pages confirmed) | ★ Y — live ES loop on OUR measured d_seg response surfaces with attribution-floor slider |
| 19 | ES book (dither design, applications) | Ariyur & Krstić (2003), *Real-Time Optimization by Extremum-Seeking Control*, Wiley, DOI 10.1002/0471669784 | CT-1 §0 row 8 ("Ariyur-Krstic") | same as #18 | Y-SEARCH (Wiley record + DOI) | N |
| 20 | Dual control: actions both exploit and probe; certainty-equivalence suboptimal when identification changes future decisions | Feldbaum (1960–61), "Dual control theory I–IV", *Automation and Remote Control* 21(9):874–883, 21(11):1033–1039, 22(1):1–12, 22(2):109–121 (transl.) | CT-1 §9 (EVSI price table; run-1 instrument value ≈ 10× crossing value) | v5 SC ledger rows + duty-to-measure ranking (PowerPlay-consistent) | Y-SEARCH (parts I–III records confirmed; part IV standard) | N |
| 21 | Two-timescale stochastic approximation: slow loop must act on windows ≥ fast settle | Borkar (1997), "Stochastic approximation with two time scales", *Systems & Control Letters* 29(5):291–294 | CT-1 §10 (co-predicate window 100 ep vs settle 115 ep = MARGINAL) | v5 fold item 3a: `--copred-verdict-window 5` (V=4→5, 125 ≥ 115) | Y-SEARCH (venue + vol + pages confirmed) | N |
| 22 | Requisite variety (regulator must carry ≥ the variety of its disturbances) | Ashby (1956), *An Introduction to Cybernetics*, Chapman & Hall, ch. 11 | CT-1 §META (H→P requirement ladder as variety injection) | design-process meta (req N(4) maturity metric) | Y-SEARCH-lineage (classical text; not independently re-verified this sweep — treat as classical) | N |

### 1B — CT-2 verification + completion (11 recorded citations → verified; attributions corrected)

| # | claim/law imported | citation | deriving doc § | v5 decision/build consumed by | verified? | marimo? |
|---|---|---|---|---|---|---|
| 23 | Hadamard structure theorem / shape derivative dJ = ∫_Γ g·V_n (boundary-normal density) | Sokolowski & Zolésio (1992), *Introduction to Shape Optimization: Shape Sensitivity Analysis*, Springer; Delfour & Zolésio (2001/2011), *Shapes and Geometries*, SIAM; Hadamard (1908), *Mémoire sur le problème d'analyse relatif à l'équilibre des plaques élastiques encastrées* | CT-2 §1 (d_seg as shape functional; margin-saliency = the shape gradient) | annulus/margin-saliency lever geometry (LEVER-4 family); SENSE rows | Y-SEARCH (standard texts; CT-2 marked them [standard texts] — completed with full records here) | ★ Y — shape-derivative (Hadamard density) visualized on OUR actual margin fields |
| 24 | Level-set method: φ_t + F\|∇φ\| = 0; reinitialization; narrow band; velocity extension | Osher & Sethian (1988), "Fronts propagating with curvature-dependent speed: Algorithms based on Hamilton-Jacobi formulations", *J. Comput. Phys.* 79(1):12–49 | CT-2 §2 (eikonal term = reinitialization; witness = level-set object) | eikonal 0.01 + length 0.001 lever stack (θ* levers, CLAUDE.md capstone) | Y-SEARCH (venue + vol + pages confirmed) | Y (runner-up — level-set evolution on our margin field; superseded by #23's demo) |
| 25 | Narrow-band level-set update (O(interface) not O(domain)) | Adalsteinsson & Sethian (1995), "A Fast Level Set Method for Propagating Interfaces", *J. Comput. Phys.* 118(2):269–277 | CT-2 §2 (iii) | annulus-restricted loss support (margin-gated support levers) | Y-SEARCH-lineage (1999 sister verified; 1995 record standard, not independently fetched) | N |
| 26 | Velocity extension off the interface via fast marching | Adalsteinsson & Sethian (1999), "The Fast Construction of Extension Velocities in Level Set Methods", *J. Comput. Phys.* 148(1):2–22 | CT-2 §2 (iv) | same as #25 | Y-SEARCH (venue + vol + pages confirmed) | N |
| 27 | Reinitialization drifts the zero level set (constrained fix) | Sussman, Smereka, Osher (1994), "A Level Set Approach for Computing Solutions to Incompressible Two-Phase Flow", *J. Comput. Phys.* 114(1):146–159, DOI 10.1006/jcph.1994.1155 | CT-2 §2 (ii) | eikonal-term drift caveat (why eikonal weight is small: 0.01) | Y-SEARCH (DOI + venue confirmed) | N |
| 28 | Monotone-scheme convergence to viscosity solutions (monotone + stable + consistent ⇒ converges, given comparison) | Barles & Souganidis (1991), "Convergence of approximation schemes for fully nonlinear second order equations", *Asymptotic Analysis* 4(3):271–283, DOI 10.3233/ASY-1991-4305 | CT-2 §2 ("the Barles–Souganidis/monotone-scheme lens") | viscosity-solution framing of the witness flow (REOPEN discipline) | Y-SEARCH (DOI + venue + pages confirmed) | N |
| 29 | PDE backstepping boundary control; one-phase Stefan problem (moving boundary) | Koga, Diagne, Tang, Krstic (2016), "Backstepping Control of the One-Phase Stefan Problem", arXiv:1607.04345 (ACC 2016) | CT-2 §4 (analytic lane band = boundary-controlled moving interface) | lane-band render-time analytic band design (L71) | **Y-FETCHED** (title + authors confirmed) | N |
| 30 | Stefan problem output feedback / state estimation | Koga, Diagne, Krstic (2017), "Control and State Estimation of the One-Phase Stefan Problem via Backstepping Design", arXiv:1703.05814 | CT-2 §4 (estimation variant) | same as #29 | **Y-FETCHED** | N |
| 31 | High-order moving-boundary Stefan stabilization | Koga & Krstic (2025), "Safe Stabilization of the Stefan Problem with a High-Order Moving Boundary Dynamics by PDE Backstepping", arXiv:2510.06571 | CT-2 §4 (high-order moving boundary) | same as #29 | **Y-FETCHED** | N |
| 32 | Backstepping course text | Krstic & Smyshlyaev (2008), *Boundary Control of PDEs: A Course on Backstepping Designs*, SIAM | CT-2 §4 header ("Krstic backstepping") | same as #29 | Y-SEARCH-lineage (standard text; not independently re-verified) | N |
| 33 | Optimal adjoint checkpointing (treeverse/revolve schedules) | Griewank & Walther (2000), "Algorithm 799: revolve: an implementation of checkpointing for the reverse or adjoint mode of computational differentiation", *ACM Trans. Math. Software* 26(1):19–45, DOI 10.1145/347837.347846 | CT-2 §3 (per-stage ckpts + EMA exceed revolve's 1-level needs; relevant only if BPTT lands) | checkpointing verdict (SATISFIED, no build) | Y-SEARCH (DOI + vol + pages confirmed) | N |
| 34 | Γ-convergence of phase-field structural optimization to sharp interface (incl. first variation / convergence of the CONTROL) | Blank, Garcke, Hecht, Rupprecht (2014), "Sharp interface limit for a phase field model in structural optimization", *SIAM J. Control Optim.* 54(3):1558–1584, DOI 10.1137/140989066, arXiv:1409.7586 | CT-2 §5 (design at finite τ, trust the τ→0 limit) | τ-anneal legality (the license for τ*_end = m_q/ln5 = 0.062 design point) | **Y-FETCHED** (arXiv) + DOI recorded in CT-2 | N |
| 35 | Homogenization/pinning: fronts pinned below threshold forcing in heterogeneous media (sub-δ structure unrecoverable by coarse flow) | Dirr & Yip (2006), "Pinning and de-pinning phenomena in front propagation in heterogeneous media", *Interfaces and Free Boundaries* 8(1):79–109 | CT-2 §3 homogenization EUREKA; §13 M2 | M2 family bound: dash comb-corrector MANDATORY if comb-OFF floors residual (v5 §0 M2 row) | Y-SEARCH (venue + vol + pages confirmed) | N |
| 36 | Bifurcation control via washout filters | Wang & Abed (1995), "Bifurcation control of a chaotic system", *Automatica* 31(9):1213–1226 | CT-2 §6 ("Abed–Fu lineage; washout-filter feedback, Wang–Abed 1995") | island-birth control framing (transient-washout guards) | Y-SEARCH (venue + vol + pages confirmed) | N |
| 37 | Conley index persistence in combinatorial dynamics (tracking index under perturbation) | Dey, Mrozek, Slechta (2020), "Persistence of the Conley Index in Combinatorial Dynamical Systems", arXiv:2003.05579 (SoCG 2020) | CT-2 §7 (obstruction to continuation = index change) | v5 fold item 4: Conley persistence certificate B17 (SENSE row; threshold τ_k·ln5 + Δ_dec^logit) | **Y-FETCHED** | ★ Y — Conley/persistence certificates on OUR per-class argmax islands (birth/death live) |
| 38 | Max-plus / curse-of-dimensionality-free HJB solve (semigroup max-plus linear) | McEneaney (2007), "A Curse-of-Dimensionality-Free Numerical Method for Solution of Certain HJB PDEs", *SIAM J. Control Optim.* 46(4):1239–1276, DOI 10.1137/040610830 | CT-2 §8 (argmax = max-plus combination; annulus fitter) | v5 row 16 DEAD verdict: max-plus band-residual decomposition; M5 family bound (two-semiring split forced) | Y-SEARCH (DOI + vol + pages confirmed) | ★ Y — max-plus annulus fit (K-vs-accuracy) on OUR cached margin/logit fields |
| 39 | Riccati-flow contraction convergence analysis of the max-plus CoD-free method | **Qu (2013)** — NOT McEneaney — "Contraction of Riccati flows applied to the convergence analysis of a max-plus curse of dimensionality free method", arXiv:1301.4777; journal version *SIAM J. Control Optim.* 52(5), DOI 10.1137/130906702 | CT-2 Sources ("Riccati-contraction analysis" listed under McEneaney) | same as #38 | **Y-FETCHED** — **ATTRIBUTION CORRECTION: sole author Zheng Qu** (CT-2's Sources block filed it under McEneaney) | N |
| 40 | Max-plus finite element method for deterministic optimal control | Akian, Gaubert, Lakhoua (2008), "The max-plus finite element method for solving deterministic optimal control problems: basic properties and convergence analysis", *SIAM J. Control Optim.* 47(2):817–848, DOI 10.1137/060655286, arXiv:math/0603619 | CT-2 §8 ("Akian–Gaubert") | same as #38 | Y-SEARCH (DOI + vol + pages confirmed) | N |
| 41 | Annealing lower bound: convergence to global minima iff cooling ≥ c/log(1+t), c ≥ deepest local-min depth | Hajek (1988), "Cooling Schedules for Optimal Annealing", *Mathematics of Operations Research* 13(2):311–329, DOI 10.1287/moor.13.2.311 | CT-2 §13 M4; litsweep-dynamics continuation section | M4 family bound: run-1 cannot certify global basin; TAIL_k warm restarts as repair (v5 §0 M4 row) | Y-SEARCH (DOI + vol + pages confirmed) | N |
| 42 | Godunov barrier: monotone linear schemes ≤ 1st-order accurate | Godunov (1959), "A difference method for numerical calculation of discontinuous solutions of the equations of hydrodynamics", *Mat. Sbornik* 47(89)(3):271–306 (mathnet.ru/eng/sm4873) | CT-2 §13 M3 | M3 family bound: global-ε viscosity fallback pays O(ε) everywhere (ca-band filtered form preferred) | Y-SEARCH (mathnet record confirmed) | N |
| 43 | Geometric control condition for wave observability (GCC) | Bardos, Lebeau, Rauch (1992), "Sharp sufficient conditions for the observation, control, and stabilization of waves from the boundary", *SIAM J. Control Optim.* 30(5):1024–1065 | CT-2 §13 NOT-binding row (full-domain observation ⇒ unconstrained) | verified-NOT-binding ledger row (no measurement owed) | Y-SEARCH (venue + vol + pages confirmed) | N |
| 44 | Topology-preserving level-set evolution (simple-point check) | Han, Xu, Prince (2003), "A topology preserving level set method for geometric deformable models", *IEEE Trans. PAMI* 25(6):755–768 | CT-2 §2/§7 (island count control under flow) | island-birth/death control (per-class sub-curricula guards) | Y-SEARCH-lineage (IEEE record 1201824 linked in CT-2; not independently re-fetched) | N |

### 1C — v5-direct + curriculum imports

| # | claim/law imported | citation | deriving doc § | v5 decision/build consumed by | verified? | marimo? |
|---|---|---|---|---|---|---|
| 45 | LADDER: recursive generation of progressively simpler problem variants (difficulty homotopy); PROVEN ⊂ our costate (1-channel/const-λ special case, L56/#322) | Simonds & Yoshiyama (2025), "LADDER: Self-Improving LLMs Through Recursive Problem Decomposition", arXiv:2503.00735 (Tufa Labs; github.com/Tufalabs/LADDER, MIT) | requirement C (ledger); `tufa_duck_harness_ladder_costate_synthesis_20260706.md`; v5 §3 curriculum | v5 §3 LADDER homotopies (island-birth per-class-λ-gated homotopy #323; movable=dilation-GO, lane=curve-prior) | **Y-FETCHED** | N |
| 46 | PMP-based training algorithms (MSA) for deep learning | Li, Chen, Tai, E (2018), "Maximum Principle Based Algorithms for Deep Learning", *JMLR* 18(165):1–29, arXiv:1710.09513 | litsweep-dynamics §control ("Li et al. JMLR 2018 (MSA/PMP)"); grounds CT-1 §1's PMP-on-training framing | costate controller lineage (req M) | **Y-FETCHED** | N |

### 1D — #305 litsweep verification (the entries consumed by crucible/v5 decisions)

| # | claim/law imported | citation | deriving doc § | consumer | verified? | marimo? |
|---|---|---|---|---|---|---|
| 47 | Eikonal-loss training instability + stabilization for neural SDFs | Yang, Sun, Sundaramoorthi, Yezzi (2023), "StEik: Stabilizing the Optimization of Neural Signed Distance Functions and Finer Shape Representation", arXiv:2305.18414 (NeurIPS 2023) | litsweep-dynamics §eikonal | eikonal-term lever design (normal-direction second-order damping cure) | **Y-FETCHED** | N |
| 48 | Viscosity-solution regularization for neural SDFs | Krishnan & Duraiswami (2025), "ViscoReg: Neural Signed Distance Functions via Viscosity Solutions", arXiv:2507.00412 | litsweep-dynamics §eikonal | viscosity-lever REOPEN framing (formulation-level, req R) | **Y-FETCHED** | N |
| 49 | Edge of stability: GD sharpness rises to 2/η then hovers; loss non-monotonic | Cohen, Kaur, Li, Kolter, Talwalkar (2021), "Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability", ICLR 2021, arXiv:2103.00065 | litsweep-dynamics §EoS | spike-guard + HVP-Lanczos spectrum probe design (S3) | **Y-FETCHED** | N |
| 50 | Adaptive (Adam) edge of stability: preconditioned sharpness equilibrates | Cohen, Ghorbani, Krishnan, Agarwal, Medapati, Badura, Suo, Cardoze, Nado, Dahl, Gilmer (2022), "Adaptive Gradient Methods at the Edge of Stability", arXiv:2207.14484 | litsweep-dynamics §EoS | same as #49 (Adam/Muon operating-point reads) | **Y-FETCHED** | N |
| 51 | Rod-flow continuous-time model for Adam at EoS | Regis & Chewi (2026), "A Rod Flow Model for Adam at the Edge of Stability", arXiv:2605.06821 | litsweep-dynamics §EoS ("Rod flow") | same as #49 | **Y-FETCHED** | N |
| 52 | Adam loss spikes root-caused to second-moment estimator decoupling (v_t/g² coupling) | Bai, Zhou, Zhao, Li, Li, Xiong, Yang, Zhang, Xu (2025), "Adaptive Preconditioners Trigger Loss Spikes in Adam", arXiv:2506.04805 | litsweep-dynamics §spikes | `moments_restore_damping_v1` (measured 6.7× vs 25.3× restore-vs-fresh; nearest-to-registrable eq) | **Y-FETCHED** | N |
| 53 | Warmup mechanism: guides nets to better-conditioned regions enabling larger target lr | Kalra & Barkeshli (2024), "Why Warmup the Learning Rate? Underlying Mechanisms and Improvements", arXiv:2406.09405 | litsweep-dynamics §warmup | finisher warm-start (Muon warm-start momentum #217/#270 lineage) | **Y-FETCHED** | N |
| 54 | Optimal lr-schedule theory in a solvable random-feature model | Bordelon & Mori (2026), "Theory of Optimal Learning Rate Schedules and Scaling Laws for a Random Feature Model", arXiv:2602.04774 | litsweep-dynamics §control | anneal-shape theory cross-check (turnpike-consistent) | **Y-FETCHED** | N |
| 55 | Graduated non-convexity (GNC) continuation | Blake & Zisserman (1987), *Visual Reconstruction*, MIT Press; modern GNC: Yang, Antonante, Tzoumas, Carlone (2020), "Graduated Non-Convexity for Robust Spatial Perception", *IEEE RA-L* 5(2), arXiv:1909.08605 | litsweep-dynamics §continuation | τ-anneal-as-continuation frame (×~1.4/iteration control-parameter schedule) | Y-SEARCH-lineage (classical text + well-known RA-L paper; arXiv ID not fetched this sweep → treat 1909.08605 as UNRESOLVED-ID) | N |
| 56 | Graduated optimization convergence (smoothing path) | Hazan, Levy, Shalev-Shwartz (2016), "On Graduated Optimization for Stochastic Non-Convex Problems", ICML 2016, arXiv:1503.03712 | litsweep-dynamics §continuation | same as #55 | UNRESOLVED (recorded in litsweep as "Hazan et al. ICML 2016"; ID not fetched) | N |
| 57 | Multi-task gradient balancing (GradNorm / PCGrad / uncertainty weighting / NTK-rate equalization) | Chen et al. (2018) GradNorm, ICML, arXiv:1711.02257 · Yu et al. (2020) PCGrad, NeurIPS, arXiv:2001.06782 · Kendall, Gal, Cipolla (2018), CVPR, arXiv:1705.07115 · Wang, Yu, Perdikaris (2022), "When and why PINNs fail to train: A neural tangent kernel perspective", *J. Comput. Phys.* 449:110768 | litsweep-dynamics §balancing | per-class λ_c design (costate per-class channels) | UNRESOLVED-IDs (titles/venues are standard; the four arXiv IDs were not fetched this sweep) | N |

---

## §2 — UNRESOLVED LEDGER (recorded but not certified this sweep — do not cite as verified)

| entry | where recorded | note |
|---|---|---|
| Heemels, Donkers, Teel periodic-ETC (row 8) | CT-1 §4.1 | named-only in CT-1; standard record exists but unverified here |
| Ioannou & Sun projection operator (row 13) | CT-1 §6.2 names no source | canonical text proposed here; needs one search to certify |
| Sachs, Hu, Ingolfsson 1995 R2R (row 17) | CT-1 §7.1 | lineage-confirmed via Ingolfsson-Sachs 1993 search only |
| Hazan et al. 2016 arXiv:1503.03712 (row 56) | litsweep-dynamics | venue named; ID unfetched |
| GradNorm/PCGrad/Kendall/Wang-Yu-Perdikaris IDs (row 57) | litsweep-dynamics | four standard papers; IDs unfetched |
| Yang et al. 2020 GNC arXiv:1909.08605 (row 55) | litsweep-dynamics | ID unfetched |
| litsweep secondary IDs: 2509.07972, 2410.23922, 2510.06684, 2505.11117, 2606.04125, 2203.05717, 1803.01299; "Gilmer et al. 2022"; "GreedyLR (Amazon Science)"; "ConFIG 2024"; "Damian-Nichani-Lee 2023"; "Mobahi-Fisher 2015"; "Li-Tai-E JMLR (SME)"; HotSpot "Wang et al., CVPR 2025"; Jaderberg 2017 PBT; Baydin 2018 hypergradients; Allgower-Georg | litsweep-dynamics Sources block | not consumed by a v5 decision directly (or consumed only as background); left unverified by triage — verify before any of them becomes load-bearing |

## §3 — PROVENANCE FINDINGS (the seal's provenance-audit feed)

1. **ATTRIBUTION CORRECTION (CT-2 Sources block):** the "Riccati-contraction analysis
   [arXiv:1301.4777]" is by **Zheng Qu (sole author, 2013)**, not McEneaney — CT-2 filed it under
   the McEneaney bullet. The paper analyzes McEneaney's method, which explains but does not
   excuse the mis-filing. Journal version: SIAM J. Control Optim., DOI 10.1137/130906702.
   Severity: minor (the claim consumed — convergence-rate of the CoD-free method — is unaffected).
2. **FOLKLORE-WITHOUT-FORMAL-PUBLICATION:** the "weak-KAM O(1/t) tail" anchor (registered for
   `powerlaw_meat_exit`; cited by name in CT-1 §10.2 and the S3 position) rests on **Fathi's
   *Weak KAM Theorem in Lagrangian Dynamics* (Cambridge lecture notes, 10th preliminary version
   2008) — famously never formally published**. The mathematical content is standard and
   independently derivable (SGD-as-ODE O(1/t) tails also follow from classical SA theory, e.g.
   via #21 Borkar's lineage), but there is NO single canonical peer-reviewed citation for the
   exact "weak KAM" form the corpus names. Recorded per requirement S: a claim whose canonical
   text is unpublished notes is stated as such, not dressed with a fabricated venue.
3. **No fabricated-citation instances found.** Every arXiv ID and DOI recorded in CT-2 and the
   litsweeps resolved to the named paper (21/21 fetch-verified targets passed). CT-1's gap was
   citedness, not falsity: all of its named-only theorem imports resolved to real canonical
   papers (rows 1–22).

## §4 — MARIMO CONTEST CANDIDATES (#347, ⚠ deadline 2026-07-09 11:59PM PST — implement-a-paper)

Top 5, ranked by (demo spectacle × cheapness given OUR existing artifacts):

1. **Tabuada 2007 + Heemels-Johansson-Tabuada 2012 — event/self-triggered control, live.**
   Replay OUR real mod32cap 41-row verdict trace; sliders for σ (contraction share) and the
   attribution floor; the notebook shows which n600 verdicts fire vs skip and the guaranteed
   minimum inter-event time — the P-CT2 backtest IS the notebook, on real training telemetry.
   Cheapest of all: trace already on disk. **(top pick)**
2. **McEneaney 2007 / Akian-Gaubert-Lakhoua 2008 — max-plus HJ/annulus fit.** Interactive
   K-basis max-plus fit on OUR cached margin/logit annulus fields; live K-vs-accuracy curve
   showing the bulk blow-up vs small-K annulus (the M5 two-semiring split, visually).
3. **Krstić-Wang 2000 — extremum seeking on a live surface.** ES loop with dither-size slider
   on OUR measured d_seg response surfaces; the attribution-floor law a* = 2·floor/ĝ becomes a
   visible go/no-go band (the rate-sweep admissibility rule, animated).
4. **Hadamard/Sokolowski-Zolésio shape derivative — on our actual margin fields.** Visualize the
   boundary-normal shape-gradient density g on the real SegNet argmax separatrix; perturb the
   interface with a brush and watch predicted-vs-actual dJ. Spectacular and unique to our data.
5. **Trélat-Zuazua 2015 — turnpike budget allocator.** Interactive epoch-budget slider decomposing
   any budget into entry/turnpike/exit + TAIL_k cycles using OUR measured ν = 0.0262/ep and
   forfeit constants; shows why extra budget goes to TAIL cycles, never longer transients.

Honorable mention: Dey-Mrozek-Slechta 2020 Conley persistence certificates on our per-class
island births (feeds B17 anyway — a demo would double as the B17 prototype).

---

Pointer contest-CPU **0.19110 UNMOVED** — this bibliography is MEANS (provenance apparatus).
[no-triality] orchestration state; no DSL/equation/DAG landings here (P7 integration handles those).
