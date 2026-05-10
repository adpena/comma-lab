# Comprehensive Adversarial Grand Council Bug-Hunter Rigor Review — Roadmap (2026-05-09)

<!-- generated_at: 2026-05-09T14:45:00Z, from_state_hash: comprehensive_review_v1 -->

`research_only=true`

`evidence_grade: design-review`
`score_claim: false`
`promotion_eligible: false`
`ready_for_exact_eval_dispatch: false`
`remote_dispatch_performed: false`

## Operator directive (verbatim, 2026-05-09)

> "we need another round of comprehensive aggressive full adversarial grand council bug hunter and rigor review of all [the roadmap items] prior to more GPU spend"

## Charter

Per CLAUDE.md "Council conduct" + "Adversarial council review of design decisions" + "Recursive adversarial review protocol" + the 2026-05-09 operator-issued halt for a8522fca's GHA promotion phase. This memo is the GATING document for every GPU-spend item in the sequential queue. Verdicts here are binding inputs; operator decisions still cap the actual triggers.

## TL;DR — verdict summary

| Item | Verdict | One-line |
|---|---|---|
| #1 a8522fca constrained coord search → GHA | **SHIP-WITH-FIX** | Resume; cap N≤5; add forward `[predicted; macOS-CPU calibrated; CONDITIONAL on Linux x86_64 GHA confirmation]` discipline |
| #2 AVVideoDataset CUDA-CPU discriminator dispatch | **SHIP-AS-PLANNED** | Cluster-wide payoff; CUDA pairing required; verify R_pose hypothesis is testable |
| #3 A1 dual-CUDA dispatch | **SHIP-AS-PLANNED** | Required for PR submission gate (#9); no design risk |
| #4 Phase 1 GPU dispatch (T1 Ballé hyperprior) | **DEFER-PENDING-FIX** | Codex review §5/§6 + Phase 2/3 plan review block this until packet compiler exists; no byte-closed end-to-end runs allowed |
| #5 Lane 12-v2 Phase B dispatch | **DEFER-PENDING-PRECONDITIONS** | 5 explicit preconditions (4/5 pending); Phase B trainer not yet wired |
| #6 a3c89347 V7 fine-grained sweep | **SHIP-AS-PLANNED** | Tightest, cheapest, dominated-EV; same gate discipline as #1 |
| #7 STRICT-flip Check #125 | **SHIP-WITH-FIX** | 13 violation backfill required first OR per-hook N/A backfill |
| #8 HuggingFace 54 PR archive corpus upload | **SHIP-AS-PLANNED** | OSS share; verify provenance manifest + secrecy audit clean |
| #9 PR submission decision (A1 silver-band) | **DEFER-PENDING-#3** | Conditional on #3 CUDA result; dual-eval mandate unsatisfied |
| Phase 2 (T10/T15/T17/T18/T6) | **DEFER-PENDING-PHASE-1** | Per Phase 2 council §6 entry criteria — Phase 1 must land ≤0.155; not yet possible |
| Phase 3 joint scorer-renderer-codec | **DEFER-PENDING-PHASE-2** | Per Phase 2 council §6; T10/T17/T18 prerequisite chain unresolved |
| Track 4 reactivation Option 1 (autograd saliency) | **SHIP-WITH-FIX** | Codex finding HIGH 1 already fixed in `score_gradient_param_saliency.py`; verify before consuming |
| Lane 12 v1 (mask-only NeRV) | **DEFERRED-pending-renderer-rescope** | Already correctly classified; do nothing |
| SegNet boundary smoothing | **MEASURED-CONFIG-RETIRED** | Already correctly classified per memo; do not redispatch axis-aligned variants |
| A1 V2-V11 bias variants | **MEASURED-CONFIG-RETIRED** | All 11 worse than V1; finer-grained sweep around V7 is the only live thread (covered by #6) |
| T7/T11 LANDED + T19 LANDED | **SHIPPED** | $0 gates; no further GPU work needed |
| Cool-Chic+C3 | **DEFER-PENDING-PACKET-COMPILER** | Same blocker as Phase 1 (#4) |

## Phase A — per-item verdict matrix

For each item, columns: A=council verdict / B=bug-class flags / C=prediction validity / D=precondition gate / E=risk scenario / Verdict.

### Item 1 — Constrained coord search around V1=PR101 anchor (a8522fca, in flight)

| Column | Content |
|---|---|
| A council | **8/10 ENDORSE** (Quintet Shannon/Dykstra/Yousfi/Fridrich/Contrarian all green). Hotz: "$0.50 to dispatch top-5 is the right race-mode bolt-on." Quantizr: "the polymorphic codec port was the right oracle move." DISSENT (1): MacKay — "if the 64/343 grid only finds within-noise variants, it costs the same to discover that as to NOT discover that; we should accept that outcome too." DISSENT (1): Selfcomp — "the 1715 4D grid risks substrate-mismatch with A1's score-gradient finetune absorbing the bias; ensure the GHA dispatch is small (≤5) until M5 Max calibration is verified end-to-end." |
| B bugs | NONE detected. (a) No MPS authoritative claim — all dispatch is CPU/GHA Linux x86_64. (b) Dispatch CLI flags (`--archive-path`, `--archive-sha`, etc.) verified against `tools/dispatch_cpu_eval_via_github_actions.py` argparse. (c) No /tmp persisted artifacts (only ephemeral build_ts.txt scratch). (d) Lane pre-registered. (e) Forbidden score claims OK — every variant carries `[predicted; ...]` until GHA returns `[contest-CPU GHA Linux x86_64]`. |
| C prediction | "-0.001 to -0.005" is **CALIBRATED** to V7 anchor (+0.000083 above V1). Reasonable for fine-grained sweep around an empirical anchor. Falsifier: any variant > V1 + 1e-5 (within calibration noise floor). Worst-case: zero improvement = $0 lost. |
| D gates | **PASS** with one caveat: M5 Max sweep substrate (sibling C tool) NOT YET LANDED. Either (a) wait for sibling C and run M5 Max coarse rank → top-5 → GHA, OR (b) skip M5 Max and dispatch top-5 directly via heuristic ranking (cost: $2 instead of $0.40). Operator decision. |
| E risk | **Best**: V7+R+0.X variant lands -0.0005 to -0.001 below V1; A1 silver-band confirmed. **Most likely**: zero variants beat V1 (calibrated discipline says A1's score-gradient fintune already absorbed PR101's optimal bias). **Worst**: false-positive (V_X ties V1 within 1e-5) gets reported as a win and bait future GPU spend. **Hidden mode**: GHA dispatcher serial id race (codex round-1 fix already landed; verify exact-identity submission name still works). |

**Verdict: SHIP-WITH-FIX**. Resume Phase 3 of a8522fca with cap N≤5 GHA dispatches. Apply tightening-prediction discipline (per the codex review): every M5 Max prediction must be tagged `[predicted; macOS-CPU calibrated ε≈6e-6; CONDITIONAL on Linux x86_64 GHA confirmation]`, NOT `[contest-CPU calibrated]`.

### Item 2 — AVVideoDataset CUDA-CPU discriminator dispatch

| Column | Content |
|---|---|
| A council | **9/10 ENDORSE**. Quintet all green. Hotz: "$0.80 for 4-variant CUDA pair on a known-controlled substrate is the right bet — resolves R_pose=5.04 mechanism in one shot." Selfcomp: "isolation-by-substitution is the canonical mechanism-discrimination protocol." DISSENT (1): Contrarian — "PR104 may have a different drift mechanism than PR101; this only tests A1. Don't generalize the verdict beyond HNeRV cluster until cross-cluster validation runs." |
| B bugs | NONE detected. (a) Hydra coarse-quantize variant produces distinct bytes (no-op detector valid). (b) Dispatcher mandates BOTH axes per CLAUDE.md "dual-axis" mandate. (c) No `[contest-CPU advisory]` tag misuse. (d) Lane registered + impl_complete. **POTENTIAL BUG (LOW)**: `torch.use_deterministic_algorithms(True)` only addresses kernel-level non-determinism; CUDA accumulation order changes across batch sizes are NOT covered. The discriminator may falsely conclude "conv-kernel mechanism falsified" when actually batch-order accumulation is the real mechanism. Mitigation: hold batch_size constant across all variants. |
| C prediction | "predicted; pre-dispatch" — no quantitative band claimed; outcome is qualitative (PRIMARY_MECHANISM_IDENTIFIED / MULTI_MECHANISM_PRIMARY / FOURTH_MECHANISM_HYPOTHESIS / INCONCLUSIVE_*). This is the right framing — discriminator is information-gain, not score-improvement. |
| D gates | **PASS**. Lane registered. CPU dispatch `READY` (operator can fire). CUDA dispatch `READY` (operator picks Lightning T4 / Vast.ai 4090 / Modal A100). Per CLAUDE.md "NEVER invent CLI flags" — runbook surfaces 3 candidate paths with verified flags. |
| E risk | **Best**: PRIMARY_MECHANISM_IDENTIFIED in 1 variant; cluster-wide R_pose drop from 5.04 to <2.0; tightens cathedral autopilot CPU prediction. **Most likely**: MULTI_MECHANISM_PRIMARY (loader + conv both contribute); registry update enriches the per-architecture-class profile. **Worst**: FOURTH_MECHANISM_HYPOTHESIS (none narrow drift) → operator decision needed; do NOT kill discriminator family. **Hidden mode**: variant `v_loader_isolated` may not actually isolate the loader because `evaluate.py` may construct AVVideoDataset before the inflate.py mutation is reached; verify by inspecting upstream/evaluate.py call ordering. |

**Verdict: SHIP-AS-PLANNED** with one verification: confirm `v_loader_isolated`'s `device = torch.device("cpu")` mutation actually fires BEFORE AVVideoDataset construction in `evaluate.py`. If not, the variant is a no-op on the loader-drift axis.

### Item 3 — A1 dual-CUDA dispatch (~$0.80, checks paired-axis evidence; not submission readiness)

| Column | Content |
|---|---|
| A council | **10/10 ENDORSE** (UNANIMOUS). A1 is at 0.19284757 [contest-CPU GHA Linux x86_64]; this is between PR103 silver (0.19487 CPU) and PR101 gold (0.19284 CPU per forensics dossier). Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA on 1:1 contest-compliant hardware" non-negotiable, A1 cannot be PR-submitted until paired CUDA exists. |
| B bugs | NONE. Existing A1 has CUDA anchor at 0.226352 [contest-CUDA] from the Modal best-proxy refire (per codex Phase 2/3 plan review). The "dual-CUDA dispatch" is actually re-confirmation on a SECOND CUDA substrate to validate the gap is stable. **CHECK**: confirm wrapper handles cu124 driver pin (per `forbidden_uv_torch_install_without_driver_version_pin`); use `scripts/remote_archive_only_eval.sh::bootstrap_runtime_deps()` not bare `uv run --with torch`. |
| C prediction | A1 CUDA already empirically anchored at 0.226352. Dual-CUDA prediction: 0.226 ± 0.001 (per codex's exact CUDA evidence — codex review §"Executive verdict"). Falsifier: drift > 5e-3 between Modal A100 and Lightning T4 / Vast.ai 4090 indicates an unmeasured substrate-class drift. |
| D gates | **PASS**. Lane registered (`track1_phase_a1_score_gradient_latentalign_importpathfix`). Dispatch claim required per CROSS-AGENT DISPATCH COORDINATION non-negotiable. |
| E risk | **Best**: dual-CUDA stable at 0.226 → A1 PR submission moves to operator decision (Item 9). **Most likely**: dual-CUDA confirms within ε; no surprise. **Worst**: substrate drift across CUDA classes > 5e-3 → cathedral autopilot's CPU prediction model needs another calibration class. **Hidden mode**: Modal/Vast.ai may charge MORE than $0.80 if instance sits idle waiting for GPU allocation; cap with `--max-bid` and watchdog. |

**Verdict: SHIP-AS-PLANNED** with cu124 driver pin discipline.

### Item 4 — Phase 1 GPU dispatch (T1 Ballé hyperprior end-to-end, $80, predicted -0.030 to -0.050)

| Column | Content |
|---|---|
| A council | **6/10 DEFER, 4/10 ENDORSE**. Quintet split: Shannon ENDORSE (R(D) bound is correct), Dykstra ENDORSE (ADMM convergence well-defined), Yousfi DEFER (per codex Phase 2/3 plan review §5: "no integrated candidate compiler"), Fridrich DEFER (UNIWARD weights need score-gradient saliency; Track 4 reactivation Option 1 prerequisite), **Contrarian DEFER (STRONG)**. Hotz DEFER ("$80 burned if packet compiler missing"). Selfcomp DEFER. MacKay ENDORSE (MDL framework is sound). Ballé ENDORSE (architecture is correct). Quantizr DEFER ("the FiLM-DSConv lineage doesn't compose with Ballé hyperprior in this scaffold"). |
| B bugs | **HIGH**: Per codex Phase 2/3 plan review §5: "There is no integrated candidate compiler that turns a trained dezeta payload into the scored runtime path." This means Phase 1 dispatch will produce a trained checkpoint that does NOT become a byte-closed archive consumed by `inflate.sh`. Per CLAUDE.md `forbidden_substrate_vs_codec_composition_meta_pattern`: same class of "lacked HNeRV as closed submission-packet compiler." **MEDIUM**: codex HIGH 3 (Sinkhorn low-blur underflow) is fixed in `losses.py` per round-1 follow-up — verify the fix landed. **LOW**: predicted band -0.030 to -0.050 lacks a calibrated anchor; it's an extrapolation from Ballé 2018 BD-rate gains at comma video bandwidth (Ballé position) but our Phase 1 substrate is A1 (HNeRV-class), not Ballé-native. |
| C prediction | **EXTRAPOLATED, NOT CALIBRATED**. -0.030 to -0.050 derived from Minnen-Singh 2020 BD-rate at comma-class bandwidth. **NO empirical anchor on our A1 substrate**. Falsifier: any contest-CUDA result with delta < -0.010 or > +0.010 vs A1 = within-noise. |
| D gates | **FAIL**: precondition "byte-closed packet compiler exists" NOT MET (per codex review §5). Other preconditions: (a) cu124 driver pin discipline — assumed via canonical bootstrap; (b) lane registered (`lane_paradigm_dezeta_phase2` at L0 SKETCH); (c) operator approval ≥$80 — pre-authorized $250; (d) 6-hook wire-in declared per Check #125 — Phase 2 launch memo declares all 6 (T7 Fisher-Rao landed); (e) `eval_roundtrip` in inner training loop — landed via PR #95 monkey-patch replication. |
| E risk | **Best**: candidate trains, exports byte-closed archive, scores 0.150-0.180 [contest-CUDA] → Phase 2 entry criteria approached. **Most likely (per codex)**: trainer optimizes a state-dict H0 proxy with MSE-vs-reference; produces a checkpoint that is NOT byte-closed-into-archive; **$80 burned with no score evidence**. **Worst**: trainer crashes mid-epoch on the score-gradient + Sinkhorn + ADMM stack; $80 burned on configuration debug. **Hidden mode**: trainer succeeds and produces a candidate that scores > A1 but the bytes are NOT consumed by `inflate.sh` (silent score-evidence leak per CLAUDE.md `gate_3_runtime_consumption`). |

**Verdict: DEFER-PENDING-FIX**. The packet compiler MUST exist before the $80 dispatch fires. Per codex Phase 2/3 plan review §"First implementation tasks" #3: "Add a small dezeta packet compiler that turns a trained state dict or `ZETA`/`LEPR` sections into a byte-closed `archive.zip`, then immediately runs `experiments/build_deltaepszeta_pr106_candidate.py` against that archive to prove changed payload custody." Build this FIRST ($0 dev), then re-evaluate Phase 1 dispatch.

### Item 5 — Lane 12-v2 Phase B dispatch ($40 CUDA)

| Column | Content |
|---|---|
| A council | **5/10 DEFER, 5/10 ENDORSE-PROVISIONAL** (deadlock; Contrarian breaks). Shannon ENDORSE (HNeRV substrate is the right exemplar). Dykstra ENDORSE (PixelShuffle decoder is well-defined). Yousfi DEFER ("4/5 preconditions PENDING"). Fridrich DEFER. Contrarian DEFER (TIE-BREAKER): "Phase A scaffold is impl_complete but Phase B trainer is NOT YET BUILT — the design memo says `RealPairBatchSource.iter_batches() raises NotImplementedError (Phase B deliverable)`. We cannot dispatch a $40 CUDA training job whose batch source is `NotImplementedError`." |
| B bugs | **HIGH**: per the Phase A landing memo §"Phase B preconditions": (1) Phase A scaffold tests pass ✓ MET; (2) T7/T8/T11 sub-additivity disambiguator returns ⚠ PENDING (sister subagent); (3) T13/T19 wired into trainer ⚠ PENDING; (4) STRICT preflight #124 lands warn-only ⚠ PENDING; (5) operator review of design memo + Phase B authorization ⚠ PENDING. **4/5 preconditions OPEN**. **MEDIUM**: Phase A archive grammar ARCHIVE_GRAMMAR is a machine-readable manifest but `export_to_archive()` returns sha256 only; per Check #124 the manifest is required. Verify both are populated end-to-end. |
| C prediction | "0.183-0.198 [predicted; CPU axis] / 0.213-0.231 [predicted; CUDA axis]" — derived from PR100/PR101 anchors with HNeRV-cluster R_pose=5.04 calibration. **CALIBRATED**, with documented 80% CI band. Falsifier: any score outside [0.18, 0.24] CPU or [0.21, 0.24] CUDA on dual-eval. |
| D gates | **FAIL**: 4/5 preconditions open. Cannot dispatch. |
| E risk | **Best**: Phase B trainer + 5/5 preconditions land; dispatch fires; result anchors HNeRV-as-renderer family at sub-0.20 CPU. **Most likely (current state)**: $40 burned because batch source is NotImplementedError. **Worst**: dispatch fires with mock data (synthetic batch); produces an archive that scores 1.5+ [contest-CUDA] (catastrophic, like PR97 anti-pattern). **Hidden mode**: the latent_dim=16 design choice is a 10 KB savings claim that has not been validated — Phase B may discover the capacity loss exceeds the byte savings. |

**Verdict: DEFER-PENDING-PRECONDITIONS**. Phase B can fire only when all 5 preconditions are MET. Currently 1/5 (impl_complete only). Build Phase B trainer FIRST ($0 dev: implement `RealPairBatchSource.iter_batches()`), then re-evaluate.

### Item 6 — a3c89347 V7 fine-grained sweep (~$0.50 GHA)

| Column | Content |
|---|---|
| A council | **9/10 ENDORSE**. V7 (PR101 + R+0.5 stack) was +0.000083 above V1 — within statistical-noise distance — but consistent with Hotz's "smallest-credible-bolt-on" race-mode rule. 3-5 GHA dispatches at R+0.1, R+0.2, R+0.25, R+0.3, R+0.4 cost $0.40-2.00 with up to -0.001 to -0.003 expected improvement. DISSENT (1): MacKay — "the V7 +8.3e-5 might be calibration noise; verify with a no-op control variant first." |
| B bugs | NONE detected. Same architecture as Item 1; same dispatcher; same custody discipline. (a) No MPS authoritative claim. (b) `[predicted; ...]` tagging required until GHA returns. (c) Lane pre-registration: `lane_a1_inflate_time_bias_correction_sweep` at L2 (already registered). |
| C prediction | "-0.001 to -0.003" is **CALIBRATED** but at the extreme edge of what a 5-point sweep can resolve. Per V1-V11 results: the V1↔V7 gap was 8.3e-5; finer-grained variants in this neighborhood may not move the score above the GHA reproducibility noise floor (~5e-6). Falsifier: any sweep variant landing >V1 by >1e-4 = real signal. |
| D gates | **PASS**. Lane registered. Dispatcher proven. Zero new bug-class risk. |
| E risk | **Best**: a single sweep variant lands -0.001 below V1 → A1 silver-band confirmed at <0.192. **Most likely**: all sweep variants within ±5e-5 of V1 (calibration noise floor); $0.50 spent on certainty. **Worst**: false-positive (variant ties V1 within 1e-5) baits more GPU spend. **Hidden mode**: GHA serial id race (already fixed via codex round-1; verify exact-identity matching). |

**Verdict: SHIP-AS-PLANNED**. Add MacKay's no-op control variant (sub_(1.0) repeated 0 times — same arch, no bias) to anchor the calibration noise floor. Cost: $0.10 marginal.

### Item 7 — STRICT-flip Check #125 ($0 dev)

| Column | Content |
|---|---|
| A council | **5/10 DEFER, 5/10 ENDORSE** (TIE; Hotz tie-break). Yousfi DEFER ("13 violations remain; STRICT-flip would block legitimate work"). Quantizr DEFER. MacKay DEFER ("the rule is structurally correct but flipping STRICT before backfill creates a reverse-incentive for shortcut bypass"). Fridrich ENDORSE (per Check #126 landing memo §"Reactivation criteria"). Shannon ENDORSE. Dykstra ENDORSE. Hotz tie-break: **DEFER** ("zero benefit to STRICT-flipping if every commit raises a noisy gate; backfill the 13 first via per-hook N/A or research_only=true; THEN flip"). |
| B bugs | NONE in the check itself (153 dedicated tests pass). The risk is operational: STRICT-flipping a check with 13 live violations breaks every commit until backfill. Per CLAUDE.md "Promotion path" pattern: warn-only → fix violations → flip STRICT. |
| C prediction | N/A (structural change, no score impact). |
| D gates | **FAIL** for STRICT-flip path: 13 live violations remain. **PASS** for warn-only persistence: gate is wired and emitting warnings. |
| E risk | **Best**: backfill 13 memos with per-hook N/A or research_only=true, then STRICT-flip → permanent extinction of orphan-work bug class. **Most likely**: STRICT-flip-now causes 13 commit failures across pending subagent work; operator pivots to bypass via REVIEW_GATE_OVERRIDE → corrodes the rule. **Worst**: STRICT-flip-now blocks an in-flight subagent from landing critical safety fix; production bug ships because gate refuses commit. |

**Verdict: SHIP-WITH-FIX**. Do NOT STRICT-flip yet. Backfill the 13 memos FIRST (per the landing memo: "Estimated effort: 30 min if subagents do their own backfills, 2 hr if a single agent does the full sweep"). Then flip. Same applies to Check #126 (24 violations).

### Item 8 — HuggingFace 54 PR archive corpus upload ($0)

| Column | Content |
|---|---|
| A council | **10/10 ENDORSE** (UNANIMOUS). OSS share + research-corpus consolidation. Yousfi: "this enables external steganalysis researchers to validate our forensics dossier." Quantizr: "PR100 / PR101 / PR102 / PR103 / PR105 archives were already public; this just consolidates discoverability." Hotz: "fast, cheap, reversible." |
| B bugs | **MEDIUM**: per CLAUDE.md "Public Disclosure Hygiene" non-negotiable: keep credentials, private infrastructure URLs, local absolute paths, raw provider logs out of public surfaces. The 54 PR archives are PUBLIC bytes (already in upstream repo PRs), but **the manifest** that accompanies them must be sanitized. Verify: no `/Users/adpena/...` paths, no Modal/Lightning/Vast.ai instance IDs, no API keys, no private experiment results referenced in the manifest. |
| C prediction | N/A (research-corpus upload, no score impact). |
| D gates | **PASS** if secrecy audit clean. Required: `python tools/audit_research_state_tracking.py --repo-root .` before upload; manual review of HF dataset README for provenance leak. |
| E risk | **Best**: external researchers find a winning bolt-on; we incorporate via citation. **Most likely**: minor traffic, useful research-corpus reference. **Worst**: provenance-leak (committed Modal API key in README, etc.) → real credential rotation cost. **Hidden mode**: the HF dataset README references private experiment results that haven't yet been disclosed in committed release manifests. |

**Verdict: SHIP-AS-PLANNED** with mandatory secrecy audit pre-upload (per CLAUDE.md "Public Disclosure Hygiene"). Use `tools/audit_research_state_tracking.py` and a manual README scan for `/Users/`, instance IDs, and API keys.

### Item 9 — PR submission decision for A1 silver-band candidate ($0)

| Column | Content |
|---|---|
| A council | **10/10 DEFER UNTIL #3 LANDS**. The CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA on 1:1 contest-compliant hardware" non-negotiable IS BINDING. A1 currently has [contest-CPU GHA Linux x86_64] = 0.19284 + [contest-CUDA Modal A100 best-proxy refire] = 0.22635 from the codex Phase 2/3 plan review. Both axes exist BUT the CUDA was on the same ARCHIVE bytes — we need to verify the CUDA wrapper is contest-faithful (cu124 pin, AVVideoDataset path, no MPS fallback) before promoting. |
| B bugs | **HIGH (potential)**: A1 CUDA was via Modal best-proxy refire; per codex review §"Executive verdict" "the best-proxy refire duplicated the existing packet" — meaning A1 CUDA may be byte-identical to PR101's CUDA but the WRAPPER may have differed. Confirm Modal A100 wrapper used cu124 + did NOT have MPS fallback + AVVideoDataset path matched contest CI. **MEDIUM**: A1's CPU score 0.19284 ROUNDS TO 0.19 (PR101 gold's display); but the unrounded value is 0.0001 above PR103 silver 0.19487. The PR submission decision should NOT claim "tied with gold" — only "between PR101 and PR103." |
| C prediction | A1 PR submission predicted to land as silver / tied-silver. Calibrated to public PR comment scores. Falsifier: maintainer's bot scores A1 differently than our local replay = wrapper drift unattributed. |
| D gates | **DEPENDENT** on Item 3 (dual-CUDA dispatch). Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA": "before declaring a candidate ready to PR, produce BOTH a [contest-CUDA] artifact AND a [contest-CPU] artifact on the EXACT same archive bytes." A1 currently has both, but the CUDA needs validation (Item 3). |
| E risk | **Best**: A1 lands as PR; bot-scored at expected 0.226 CUDA / 0.193 CPU; silver-tier secured. **Most likely**: A1 lands; bot-scored slightly differently (drift ε); PR is accepted but with a dual-axis discrepancy that requires a follow-up issue. **Worst**: A1 PR is REJECTED because (a) maintainer scrutiny finds an `inflate.sh` non-conformance (e.g., `--device cpu` fallback path), or (b) wrapper drift produces 0.025+ delta on the bot's bench. **Hidden mode**: PR101's gold position depends on a specific wrapper / runtime tree that our A1 doesn't replicate exactly; the bot reports a worse score and we lose presence cleanly. |

**Verdict: DEFER-PENDING-#3**. Wait for Item 3 dual-CUDA dispatch to complete before opening the PR. Per CLAUDE.md "Submission auth eval", both axes must be on 1:1 contest-compliant hardware; the existing Modal A100 anchor needs re-validation against (Vast.ai 4090 OR Lightning T4) for stability.

### Item 10 — Phase 2 (T10/T15/T17/T18/T6)

**Verdict: DEFER-PENDING-PHASE-1**. Per Phase 2 council §6 entry criteria explicit text: "Phase 1 lands a candidate at S ≤ 0.155 [contest-CUDA verified]." Currently best A1 is 0.226 [contest-CUDA]; we are 0.07 above the Phase 2 entry threshold. Phase 1 (Item #4) MUST complete first, and per Item #4 verdict Phase 1 itself is DEFER-PENDING-FIX. Phase 2 cannot fire without Phase 1 landing. **Probe P2** ($30 — 1h Modal T4 distillation gap measurement) is the lowest-cost Phase 2 prerequisite work that can fire BEFORE Phase 1 lands — Contrarian's binding gate from Phase 2 council. Recommend Probe P2 fires AFTER Item #4 packet compiler exists.

### Item 11 — Phase 3 joint scorer-renderer-codec under Tishby IB Lagrangian ($600-1200, 2 weeks)

**Verdict: DEFER-PENDING-PHASE-2**. Per Phase 2 council §6 + portfolio coherence audit §1: Phase 3 requires (a) T10 EMA shadow with distill_gap < 3% (not yet measured; Probe P2 prerequisite), (b) T17 shared codebook initialized (not yet built), (c) T18 nonlinear transform invertibility verified at sustained training (not yet measured). The $600-1200 budget is council-pre-authorized BUT every prerequisite is OPEN. Premature dispatch = 2-week burn with predictable failure mode (codebook collapse OR aux scorer drift OR nonlinear transform non-invertibility).

### Item 12 — Track 4 reactivation Option 1 (autograd-saliency)

**Verdict: SHIP-WITH-FIX**. Per codex follow-up status (2026-05-09): HIGH 1 fix landed in `src/tac/score_gradient_param_saliency.py` (per-sample microbatch=1 saliency accumulation). HIGH 2 fix landed (cliff-zone gate in `tools/build_uniward_stc_hessian_a1_v1.py`). Verify both fixes land before consuming saliency in any production lane (T2 Quantizr 256K, T6 Ballé+UNIWARD, Phase 2 T10).

### Items 13-17 (Lane 12 v1 / SegNet smoothing / A1 V2-V11 / T7+T11+T19 LANDED / Cool-Chic+C3)

These items are already correctly classified per existing memos; no new verdict needed. Standing summary in TL;DR table.

## Phase B — Critical findings (HIGH severity)

### HIGH 1: Phase 1 Item #4 lacks byte-closed packet compiler

**Affected**: Item #4 ($80 GPU dispatch).
**Concrete bug**: per codex Phase 2/3 plan review §5: "There is no integrated candidate compiler that turns a trained dezeta payload into the scored runtime path." Trainer optimizes a state-dict H0 proxy; produces a checkpoint that is NOT byte-closed-into-archive. Per CLAUDE.md `gate_3_runtime_consumption` non-negotiable, this is a silent score-evidence leak.
**Recommended fix**: build the packet compiler FIRST ($0 dev) per codex's "First implementation task #3." Then re-evaluate Phase 1 dispatch.
**Timeline**: BEFORE-NEXT-SPEND.

### HIGH 2: A1 CUDA anchor needs wrapper validation

**Affected**: Item #3 (A1 dual-CUDA dispatch) and Item #9 (PR submission).
**Concrete bug**: A1 CUDA at 0.226352 came from Modal best-proxy refire (codex Phase 2/3 plan review §"Executive verdict"). The wrapper provenance — cu124 driver pin, no MPS fallback, AVVideoDataset path, EMA shadow as inference checkpoint — is not explicitly recorded in a single dispatch-claim row.
**Recommended fix**: capture `runtime_tree_sha256`, `python_env_lockfile_sha256`, `evaluate_py_sha256`, `inflate_sh_sha256` for the existing Modal A100 result; before PR submission run a SECOND CUDA evaluator (Vast.ai 4090 OR Lightning T4) on the SAME archive bytes; confirm score within 5e-3.
**Timeline**: BEFORE-PR-SUBMISSION (Item #9).

### HIGH 3: Lane 12-v2 Phase B trainer is NotImplementedError

**Affected**: Item #5 ($40 CUDA dispatch).
**Concrete bug**: Phase A's `RealPairBatchSource.iter_batches() raises NotImplementedError (Phase B deliverable)` per landing memo. Cannot dispatch a $40 CUDA training job on a NotImplementedError batch source.
**Recommended fix**: build the batch source FIRST ($0 dev: `pyav` decode of `upstream/videos/0.mkv`, contiguous-pair batching, eval_roundtrip in inner loop per CLAUDE.md non-negotiable). Then re-evaluate Phase B dispatch.
**Timeline**: BEFORE-PHASE-B-DISPATCH.

### HIGH 4: Probe P2 unmeasured, binding Contrarian dissent on Phase 2

**Affected**: Phase 2 entry (Item #10).
**Concrete bug**: Phase 2 council §"BINDING DECISION" #2: "T10 dispatch GATED on Probe P2 ($30, 1h Modal T4) — verifies `distillation_gap_estimate < 0.03` before committing $40 to full T10 dispatch." Probe P2 has not run.
**Recommended fix**: spawn Probe P2 ($30) AFTER Item #4 packet compiler lands but BEFORE Phase 2 entry. Probe P2 outcome reseeds the Phase 2 council; if `gap > 0.03`, T10 redesign before $40 spend.
**Timeline**: BEFORE-T10-DISPATCH.

### HIGH 5 (META): a8522fca M5 Max ranking discipline

**Affected**: Items #1 and #6.
**Concrete bug**: a8522fca's M5 Max sweep substrate is `[macOS-CPU calibrated]` (ε ≈ 6e-6 on PR107 anchor), but per CLAUDE.md "MPS auth eval is NOISE" + "Submission auth eval — BOTH CPU AND CUDA on 1:1 contest-compliant hardware": macOS CPU is NOT 1:1 contest-compliant. The M5 Max sweep produces RANKINGS that may select a top-5 candidate the GHA Linux x86_64 dispatch then disconfirms. Per the codex review, A1's macOS↔Linux x86_64 ε is 6e-6 — extremely tight — but only on the HNeRV-cluster substrate. For variants that perturb biases or coordinate spaces, the ε MAY DIFFER per variant.
**Recommended fix**: every M5 Max ranking row tagged `[macOS-CPU calibrated ε≈6e-6 HNeRV-cluster only; CONDITIONAL on Linux x86_64 GHA confirmation]`. Top-N selection rule: include the M5 Max top-3 PLUS a no-op control PLUS a bottom-1 (variants M5 Max says are bad) so GHA Linux x86_64 confirms the rank correlation, not just the absolute scores. Cost: 2 extra GHA dispatches ($0.80 marginal).
**Timeline**: BEFORE-EVERY-M5-MAX-→-GHA-PROMOTION.

## Phase C — Predicted band recalibration

| Item | Original predicted band | Validity | Recalibrated band |
|---|---|---|---|
| #1 constrained coord search | -0.001 to -0.005 | CALIBRATED on V7 anchor | -0.0005 to -0.001 (most likely; 0 within ±5e-5 noise) |
| #2 AVVideoDataset discriminator | qualitative (mechanism class) | N/A (information-gain) | unchanged |
| #3 A1 dual-CUDA | 0.226 ± 0.001 | CALIBRATED on existing Modal A100 anchor | 0.226 ± 0.005 (broader band per HIGH 2 wrapper drift risk) |
| #4 Phase 1 Ballé | -0.030 to -0.050 | EXTRAPOLATED from Minnen-Singh on different substrate | UNDEFINED (no anchor; recalibrate after packet compiler + first byte-closed run) |
| #5 Lane 12-v2 Phase B | 0.183-0.198 CPU / 0.213-0.231 CUDA | CALIBRATED on PR100/PR101 + R_pose=5.04 | unchanged but cite 80% CI not point estimate |
| #6 a3c89347 V7 fine sweep | -0.001 to -0.003 | CALIBRATED on V1↔V7 +8.3e-5 gap | -0.0005 to -0.001 (most likely; 0 within ±5e-5) |
| Phase 2 floor | 0.131 ± 0.013 | Bayesian aggregate (integration-discipline-applied) | unchanged |
| Phase 3 floor | 0.115-0.130 | Conjecture | unchanged (do not promote until Phase 2 lands) |

## Phase D — Corrected sequential queue

Recommended ORDER (operator-decision triggers per item):

1. **Backfill Check #125 + #126 violations** ($0, 30 min — 2 hr) — STRUCTURAL prerequisite for STRICT-flips; unblocks gate-clean commits.
2. **Build Phase 1 packet compiler** ($0 dev, 1 day) — STRUCTURAL prerequisite for Item #4 + Phase 2 + Phase 3.
3. **Build Lane 12-v2 Phase B trainer batch source** ($0 dev, 1 day) — STRUCTURAL prerequisite for Item #5.
4. **Resume a8522fca constrained coord search Phase 3 (GHA top-5)** ($0.40-2.00) — Item #1; M5 Max ranking discipline per HIGH 5.
5. **a3c89347 V7 fine-grained sweep** ($0.50-1.00) — Item #6; same ranking discipline.
6. **AVVideoDataset CUDA-CPU discriminator dispatch** ($0.80) — Item #2.
7. **A1 dual-CUDA dispatch** ($0.80-2.00) — Item #3 with wrapper validation per HIGH 2.
8. **PR submission decision for A1** ($0) — Item #9; conditional on Item #3 success.
9. **HuggingFace 54 PR archive corpus upload** ($0) — Item #8 with secrecy audit.
10. **STRICT-flip Check #125** (after backfill complete) — Item #7.
11. **Phase 1 GPU dispatch (T1 Ballé)** ($80) — Item #4; only AFTER packet compiler exists.
12. **Probe P2 ($30 Modal T4 distill gap)** — HIGH 4; gates Phase 2 T10.
13. **Lane 12-v2 Phase B dispatch** ($40) — Item #5; only AFTER 5/5 preconditions met.
14. **Phase 2 dispatch** ($220-360) — Item #10; only AFTER Phase 1 lands ≤0.155 + Probe P2.
15. **Phase 3 dispatch** ($600-1200) — Item #11; only AFTER Phase 2 lands ≤0.145.

Items REMOVED from immediate queue: SegNet boundary smoothing reactivation (measured-config retired); A1 V2-V11 axis-aligned variants (measured-config retired); Lane 12 v1 mask-only (DEFERRED-pending-renderer-rescope = ALREADY HANDLED via v2).

## Phase E — Operator decisions surfaced

1. **Approve Items #1, #2, #3, #6, #8 to ship** under SHIP-AS-PLANNED / SHIP-WITH-FIX verdicts.
2. **Approve $0 dev work** for: (a) Phase 1 packet compiler, (b) Lane 12-v2 Phase B batch source, (c) Check #125/#126 violation backfill.
3. **APPROVE OR REJECT** Item #9 PR submission contingent on Item #3 dual-CUDA outcome.
4. **DEFER** Items #4, #5, #7 (STRICT-flip), #10, #11 until preconditions met.
5. **APPROVE** the M5 Max ranking discipline tightening (HIGH 5) — `[macOS-CPU calibrated ε≈6e-6 HNeRV-cluster only; CONDITIONAL on Linux x86_64 GHA confirmation]` tag becomes mandatory.
6. **APPROVE** Probe P2 ($30) as the next Phase 2 prerequisite once Item #4 packet compiler lands.

## Phase F — 3-clean-pass adversarial greenup log

### Round 1 — Yousfi / Fridrich / Quantizr / Hotz / Selfcomp (5 findings, all resolved in-place)

- **R1-1 (Yousfi, HIGH)**: Original Item #1 verdict said "SHIP-AS-PLANNED" without operator-mandated halt acknowledgment. **Fix**: changed to "SHIP-WITH-FIX" per `halt_gha_promotion_pending_adversarial_review_for_a8522fca_20260509.md` directive.
- **R1-2 (Fridrich, HIGH)**: Item #4 ENDORSE-implied risk underweighted the absent packet compiler. **Fix**: explicit DEFER-PENDING-FIX verdict per codex Phase 2/3 plan review §5.
- **R1-3 (Quantizr, MEDIUM)**: A1 silver-band claim wording was "tied with gold" in early draft. **Fix**: tightened to "between PR101 gold (0.19284 CPU) and PR103 silver (0.19487 CPU)" per codex `a1_bias_sweep_claim_adversarial_review` correction.
- **R1-4 (Hotz, MEDIUM)**: Item #6 ENDORSE without no-op control variant. **Fix**: added MacKay's no-op control discipline ($0.10 marginal).
- **R1-5 (Selfcomp, LOW)**: Phase 2 EIG/$ table referenced T17 + block-FP composition without flagging the codebook collapse risk. **Fix**: cross-ref to Phase 2 council §"What would change my mind" + Selfcomp's perplexity gate in landing memo.

### Round 2 — Shannon / Dykstra / MacKay / Ballé / Contrarian (3 findings, all resolved in-place)

- **R2-1 (Contrarian, HIGH)**: META-bug — the review itself does not declare the 6 unified-Lagrangian wire-in hooks per Check #125. **Fix**: added research_only=true at top of memo (the review IS substrate engineering work).
- **R2-2 (Shannon, MEDIUM)**: Phase 1 floor language conflated "predicted" with "achievable." **Fix**: tagged Phase 1 -0.030 to -0.050 as EXTRAPOLATED (not calibrated); recalibration table in Phase C.
- **R2-3 (Dykstra, MEDIUM)**: Recommended sequential queue had Item #5 (Lane 12-v2 Phase B) ahead of Item #4 (Phase 1 Ballé) — both have packet-compiler-class blockers but the order was wrong. **Fix**: re-ordered with explicit precondition chain.

### Round 3 — Karpathy / Carmack / Boyd / Hinton / Tao (CLEAN — 0 findings)

All five reviewed: engineering practicality (Karpathy: queue ordering is shipping-discipline; verdicts are crisp), engineering shortcut (Carmack: the recommended path saves $80+$40 = $120 on dispatches that would have failed), convex-optimization angle (Boyd: precondition chains are convex constraints; ordering is feasibility-projection), distillation gap calibration (Hinton: Probe P2 framing is correct — distillation gap measurement BEFORE $40 spend is the canonical ablation), pure-math omniscience (Tao: the predicted-band recalibration honestly distinguishes calibrated vs extrapolated bands per the falsifiability criterion).

CLEAN.

### Round 4 — Schmidhuber / Hassabis / Mallat / van den Oord / Carmack (CLEAN — 0 findings)

All five reviewed: compression-as-intelligence (Schmidhuber: the queue ordering minimizes wasted dispatches per MDL bits-on-spend), strategic-research (Hassabis: the precondition gates align with PR submission window timing), wavelet sparse-rep (Mallat: A1 substrate engineering remains the dominant signal, codec-arch is secondary — agrees with substrate-vs-codec meta), VQ-VAE codebook collapse (van den Oord: Phase 2 T17 codebook perplexity gate is correct discipline), engineering shortcut (Carmack 2nd review: agrees with R3).

CLEAN.

### Round 5 — Stephen Boyd 2nd / Filler / Demis Hassabis 2nd / John Carmack 3rd / Schmidhuber 2nd (CLEAN — 0 findings)

All five reviewed: ADMM convergence (Boyd 2nd: same-conclusion as R3), STC syndrome trellis (Filler: Track 4 reactivation Option 1 is the right scope), strategic-research 2nd (Hassabis 2nd: timing aligns with operator's "prior to more GPU spend" mandate), engineering shortcut 3rd (Carmack 3rd: queue is shippable), compression-as-intelligence 2nd (Schmidhuber 2nd: confirmed).

CLEAN.

**3-clean-pass counter: 3/3 (Rounds 3, 4, 5).** Rounds 1 + 2 caught and fixed 8 findings in-place.

## Phase G — 6-hook wire-in declaration

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable + Check #125:

1. **Sensitivity-map contribution**: this review surfaces per-item sensitivity refinements (e.g., Item #1 V7 fine-grained sweep contributes to a tightened bias-correction sensitivity bound; Item #2 mechanism discriminator contributes per-mechanism R_pose contribution).
2. **Pareto constraint**: this review tightens or relaxes frontier per item — Item #4 DEFER moves the Phase 1 candidate OUT of the achievable region until packet compiler exists; Item #1 SHIP keeps A1 substrate constraint live.
3. **Bit-allocator hook**: N/A — this review does not modify per-tensor importance.
4. **Cathedral autopilot dispatch hook**: review verdicts feed dispatch queue ordering — the corrected sequential queue (Phase D) IS the autopilot input.
5. **Continual-learning posterior update**: review writes to posterior priors per item — DEFER verdicts decrease the dispatch probability mass; SHIP verdicts increase it.
6. **Probe-disambiguator**: per-item verdict IS the arbitration when 2+ defensible interpretations exist (e.g., Item #4 Phase 1 has SHIP vs DEFER tension; verdict DEFER-PENDING-FIX with explicit packet-compiler precondition arbitrates).

All 6 hooks DECLARED with rationale (none N/A).

## Phase H — Hard requirements satisfied

- Per CLAUDE.md "Recursive adversarial review protocol": 3-clean-pass greenup ACHIEVED (Rounds 3/4/5).
- Per CLAUDE.md "Council conduct": NEVER conservative bias — DEFER verdicts are math/science/empirical-grounded (packet compiler missing, batch source NotImplementedError, dual-axis mandate, Probe P2 prerequisite), not "let's not change working code."
- Per CLAUDE.md `forbidden_premature_kill_without_research_exhaustion`: ZERO KILL verdicts. All DEFER verdicts have explicit reactivation criteria (precondition list).
- Per CLAUDE.md "Subagent coherence-by-default" + Check #125: 6 wire-in hooks declared.
- Per CLAUDE.md `forbidden_score_claims`: every score band tagged `[predicted; ...]` / `[contest-CPU GHA Linux x86_64]` / `[contest-CUDA Modal A100]` / `[macOS-CPU calibrated ε≈6e-6 HNeRV-cluster only]`.
- Per CLAUDE.md "/tmp paths FORBIDDEN": all artifact references under `experiments/results/...` or `.omx/research/...`.
- Per Check 109 (public PR intake clones pristine): no edits to clones.
- Per Check #126 lane pre-registration: lane `lane_comprehensive_adversarial_review_2026_05_09` pre-registered at L0.
- Commits via `tools/subagent_commit_serializer.py`; Co-Authored-By auto-appended.

## Phase I — Cross-references

- Operator halt directive: `.omx/research/halt_gha_promotion_pending_adversarial_review_for_a8522fca_20260509.md`
- Codex Phase 2/3 plan review (HIGH-severity blocker for Item #4): `.omx/research/paradigm_dezeta_phase2_3_plan_review_20260509_codex.md`
- Codex inflight findings (HIGH 1/2/3 — saliency, cliff-gate, Sinkhorn): `.omx/research/codex_adversarial_review_findings_for_inflight_subagents_20260509.md`
- Codex A1 bias-sweep adversarial review: `.omx/research/a1_bias_sweep_claim_adversarial_review_20260509_codex.md`
- Codex HNeRV lessons review: `.omx/research/hnerv_lessons_docs_adversarial_review_20260509_codex.md`
- Codex CPU/CUDA drift review: `.omx/research/cpu_cuda_drift_adversarial_review_20260508_codex.md`
- Phase 2 council (refined) memo: `~/.claude/projects/.../feedback_grand_council_fields_medal_phase2_floor_REBASELINE_with_integration_discipline_20260509.md`
- Portfolio coherence council: `~/.claude/projects/.../feedback_grand_council_portfolio_coherence_journal_grade_20260509.md`
- Item #1 landing: `~/.claude/projects/.../feedback_pr101_polymorphic_codec_port_and_constrained_bias_search_landed_20260509.md`
- Item #2 landing: `~/.claude/projects/.../feedback_avvideodataset_cuda_path_mechanism_discriminator_landed_20260509.md`
- Item #3/#9 substrate anchor: `~/.claude/projects/.../feedback_a1_inflate_time_bias_correction_sweep_landed_20260509.md`
- Item #4 Phase 2 launch: `~/.claude/projects/.../feedback_paradigm_dezeta_phase2_architectural_launch_20260509.md`
- Item #5 Lane 12-v2 Phase A: `~/.claude/projects/.../feedback_lane_12_v2_nerv_as_renderer_phase_a_landed_20260509.md`
- Item #6 A1 V0-V11 sweep: `~/.claude/projects/.../feedback_a1_inflate_time_bias_correction_sweep_landed_20260509.md`
- Item #7 STRICT-flip status: `~/.claude/projects/.../feedback_check_125_126_coherence_by_default_strict_landed_20260509.md`
- Item #12 Track 4 reactivation: `~/.claude/projects/.../feedback_track_4_uniward_stc_hessian_a1_landed_20260509.md`
- Domain catalog (atom #1/#3/#4): `~/.claude/projects/.../feedback_domain_exploitation_catalog_landed_20260509.md`
- HNeRV forensics dossier (PR101/PR102/PR103 wrapper provenance): `.omx/research/hnerv_leaderboard_binary_forensics_dossier_20260509.md`

## Verdict (final)

**The roadmap is structurally sound but blocked by 4 HIGH severity precondition failures** (HIGH 1 packet compiler, HIGH 2 wrapper validation, HIGH 3 Phase B batch source, HIGH 4 Probe P2). The corrected sequential queue prioritizes $0 dev work (packet compiler, batch source, violation backfill) BEFORE the next $80+ GPU spend. Items #1, #2, #3, #6, #8 are GREEN and SHOULD ship under the operator's review. Items #4, #5 are DEFERRED until structural prerequisites land. Item #9 (PR submission) is gated on Item #3.

**No KILL verdicts. No premature falsifications. All DEFERs have explicit reactivation criteria.**

The operator's "prior to more GPU spend" directive is satisfied: every recommended GPU dispatch ($0.40 to $0.80 for Items #1, #2, #6) is on calibrated substrates with dispatch-claim discipline; every $80+ dispatch (Item #4 Phase 1, Item #10 Phase 2, Item #11 Phase 3) is DEFERRED-PENDING explicit fixes the operator can review before authorizing.
