---
arm: ddm_ar5
title: "arXiv 2608.03353 Markov-chain convergence crosswalk"
utc: 2026-08-05
research_only: true
score_claim: false
promotion_eligible: false
pointer_moved: false
axis: "[paper-crosswalk scorer-free]"
tokens: "[no-triality] [p0-ledger-ok]"
---

# AR5 Receipt - arXiv 2608.03353 Crosswalk

## Answer First

arXiv `2608.03353v1` is **A Direct Route to Markov Chain Convergence via Asymptotic Equivalence with the Target** by **Patrick Forre**. arXiv reports v1 submitted 2026-08-04.

Verdict for Pact: **ADOPT the paper only as a scorer-free convergence-certification lens for stochastic/adaptive search and sampling surfaces.** Its useful transfer is the two-sided rule: do not call an iterative or sampled process converged unless (1) starting-state residue is washing out and (2) the process can see the target support it is supposed to cover. This reinforces TJ1's cap-bound-not-converged law, m88/m96 selection hygiene, and SL1/TP1 tail/readback requirements. It is **not** a renderer, pose solver, token coder, loss function, archive grammar, or score mechanism.

No scorer, no launch, no archive mutation, no `upstream/evaluate.py`, and no frontier number moved.

## Paper Identification

Sources attempted/fetched:

- `https://arxiv.org/abs/2608.03353` - browser fetch succeeded.
- `https://arxiv.org/html/2608.03353` - browser fetch succeeded and was deep-read.
- `https://arxiv.org/pdf/2608.03353` / `2608.03353v1.pdf` - browser PDF/source did not return usable custody in this turn; local `curl` failed with DNS resolution error for `arxiv.org`.

| field | observed value |
|---|---|
| arXiv id | `2608.03353v1` |
| title | A Direct Route to Markov Chain Convergence via Asymptotic Equivalence with the Target |
| author | Patrick Forre |
| affiliation shown | AI4Science Lab; Korteweg-de Vries Institute for Mathematics; University of Amsterdam |
| submitted | 2026-08-04 |
| subjects | Probability; Machine Learning; Statistics Theory; Computation; Machine Learning |
| arXiv size | 92 KB |
| paper claim | A short self-contained route to standard Markov-chain convergence results via a necessary/sufficient asymptotic-equivalence criterion, with applications to positive-density kernels, Metropolis-Hastings, eventually-positive kernels, random-scan Gibbs, parallel tempering, and ergodic averages. |

## Deep Read

The paper reframes total-variation convergence of a Markov kernel `T` with invariant target `pi` through the Lebesgue decomposition of each started law `T^n_x` with respect to `pi`. Two asymptotic properties are load-bearing:

- **Asymptotic absolute continuity:** the part of `T^n_x` singular to `pi` vanishes for every starting point.
- **Asymptotic domination of the target:** for `pi`-almost every starting point, the part of `pi` invisible to `T^n_x` vanishes.

On countably generated measurable spaces, those two conditions characterize convergence to the unique invariant probability measure. The density-form proof uses a jointly measurable positive minorant plus regularization: overlap gives contraction for dominated/flat laws; asymptotic absolute continuity brings arbitrary initial laws into that dominated class. The paper emphasizes that strict finite-time absolute continuity is not required; Metropolis-Hastings can keep an atom at the starting point, and random-scan Gibbs can be singular at every finite step.

The paper also explicitly refuses what it cannot provide: it gives qualitative convergence, not a usable convergence rate. Even strictly positive transition densities need not give a uniform rate on general spaces. The ergodic-average corollary is derived separately through Birkhoff, not from marginal law convergence alone.

## Ranked Crosswalk

| rank | disposition | paper element | Pact surface | label | named consumer | falsifier / stop rule | cost |
|---:|---|---|---|---|---|---|---|
| 1 | ADOPT | Two-sided convergence criterion: residue must wash out and target support must be seen. | TJ1 trajectory stopping, SQ2/SL1 cap-bound reads, terminal pose/free-frame tail receipts. | DERIVED from paper plus MEASURED local cap-censoring need. | `trajectory_derived_stopping_law_v1`, `tools/replay_tj1_trajectory_stopping.py`, SL1 receipt reader. | If the procedure is a deterministic exact solve with a complete optimality certificate, classify N-A. If existing receipts already print both tail/residue mass and target-coverage support for every item, reclassify ALREADY-EMBODIED. A safety cap remains `safety_bound_REPORTED`, never convergence. | $0 receipt/schema audit over existing traces. |
| 2 | ADOPT | Persistent atoms can be legal, but only if their mass decays. | Terminal 6-eq pose solve and free-frame pose upper-bound tails. | CONJECTURE process transfer; no pose measurement. | SL1 terminal-pose/free-frame tail confirm; `terminal_pose_gn` status readers. | If tail-pair mass or terminal-state occupancy does not shrink under deeper budget or relinearization on existing logs, this lens folds for pose. If it shrinks but has no rate, it is still not a stop certificate. | $0 replay/readback on current logs only. |
| 3 | ALREADY-EMBODIED / reinforce | Qualitative convergence gives no rate. | TJ1, SQ2, cap-bound solver loops, budget ladders. | MEASURED locally; paper supports the doctrine. | MAIN harvest; SQ2/SL1/jd1 readouts. | Any row that says "converged" from a fixed iteration cap or a monotone partial prefix fails; use projected tail gain in S-units or report floor-only. | Already active; no new work. |
| 4 | ADOPT | Add a small global/positive proposal to prevent unreachable regions, but do not mistake it for a rate guarantee. | Adaptive candidate search, token/correction proposal generators, stratified reraces. | DERIVED analogy. | EN1 context race, future token/correction search drivers, `gr1` stratified rerace. | Candidate generator must beat same-coder byte preflight or improve cached support coverage before any scorer spend. On `IX2TOK01`, SV2/XO1 still block semantic/orderer byte claims unless match structure improves. | $0 candidate/support preflight. |
| 5 | ADOPT | "For every x" is load-bearing in Gibbs-style updates. | Stratified sampling and verdict hygiene. | MEASURED local need from m88/m96/P4X/NA2. | `subset_selection`, `subset_selection_gate`, GR1, TP1/SL1 sampled reads. | Any receipt silent on selection mode is treated as video-order prefix. Local conclusions need pair ids, seed, denominator, support labels, and residual spread; otherwise the verdict is under-specified. | $0 metadata/readback pass. |
| 6 | ALREADY-EMBODIED / reinforce | Common minorant/overlap is the convergence object, not a local-mode improvement. | Correction stacks under joint Seg/Pose/rate pricing. | MEASURED local governance plus paper support. | R8 composed-stage gate, SL1 correction-stack reader, VW1/verdict wiring. | A correction that improves one axis but destroys pose bank or bytes remains non-adoptable. Existing examples: rt1/fz4-style map repair and SQ1/SQ2 pose-collateral reads. | Already active; keep enforcing. |
| 7 | N-A / FOLDED for bytes | Markov-chain convergence theorem. | Token coder races and archive bytes. | MEASURED local blocker. | EN1, future token-byte preflights. | The paper is not a coder. A Markov-context coder must beat shipped `IX2TOK01`/Brotli-Q11 bytes on the same object and improve LZ match structure; `#859`/SV2 currently block base-rule/symbol-order claims. | No action now. |
| 8 | N-A for loss design | The proof has no differentiable scorer objective or optimizer update. | Margin-hinge / tau_softplus loss engineering. | DERIVED mismatch. | EN1 margin-hinge-to-tau build only as unaffected live work. | If cited as a new loss or gradient mechanism, refuse. The real debt is wiring and matched A/B through the live trainer. | $0 closed. |
| 9 | N-A for score movement | Uniqueness/ergodic-average theorem. | Contest score, archive promotion, pointer update. | DERIVED mismatch. | none | Reopen only with an executable Pact stochastic sampler whose stationary target is explicitly a contest-S objective and whose output is a byte-closed candidate. | $0 closed. |

## Zero-Dollar Probes

| probe | disposition | fire order | close/falsify condition |
|---|---|---|---|
| AR5-TWO-SIDED-STOP-READBACK | QUEUED-WITH-FIRE-ORDER | On the next SL1/SQ2/TJ1-style read, print for each iterative item: start-residue/tail state, target-support coverage, stop reason, and projected remaining S gain if available. | Adopt if it changes a convergence/floor classification; fold if every live receipt already carries these fields. |
| AR5-TAIL-MASS-POSE-REPLAY | QUEUED-WITH-FIRE-ORDER | Use existing SL1/terminal-pose/free-frame traces only; measure whether tail-pair/terminal occupancy decays with budget or relinearization. | Fold if tail mass is nondecaying or unrelated to final pose residual. If decays without rate, keep floor-only wording. |
| AR5-POSITIVE-MIXTURE-CANDIDATE-COVERAGE | QUEUED-WITH-FIRE-ORDER | For an existing byte-only token/correction candidate set, add a deterministic low-frequency/global proposal bucket and report support coverage plus same-coder bytes. | Fold unless support improves and same-coder bytes do not lose to shipped `IX2TOK01`/Brotli-Q11. No scorer. |
| AR5-SAMPLING-METADATA-GATE | ALREADY-EMBODIED / keep enforcing | Every sampled read prints pair ids, seed, selection mode, denominators, support labels, and residual spread. | Any missing field downgrades the verdict to scoped/under-specified. |

## RECALL EVIDENCE

| source searched | query or lookup | found beyond charter seeds | impact |
|---|---|---|---|
| `.omx/tmp/codex_runs/ar5_prompt.md`, `_common_contract.md` | direct read | AR5 is paper-crosswalk only; no scorer/no launch; deliver `AR5_RECEIPT.md`; serializer commit with `[no-triality] [p0-ledger-ok]`; final own frontier line fixed. | Kept scope to one markdown receipt and no scorer work. |
| `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md` | required governing reads | Own-vehicle line is `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`; contest pointer borrowed/unmoved; TP1 birth A/B burning; SL1/EN1/JD1 live; no scorer slot for AR5. | Ranked only scorer-free probes and avoided score language. |
| arXiv browser fetch | `2608.03353` abstract/html/PDF/source attempts | Paper identified as Forre Markov-chain convergence note; HTML accessible; PDF/source custody not established; local `curl` DNS failed. | Receipt uses abstract/html custody only and states PDF limit. |
| paper/checking discipline | `papers_checked`, `papers-checked`, `m44`, `L55`, prior AR receipts | Found AR2/AR4 receipt pattern and paper-check discipline; no prior AR5 receipt found in scoped search. | Used answer-first receipt, recall table, zero-dollar probes, and bounded negative wording. |
| canonical equations registry | `.venv/bin/python tools/list_canonical_equations.py --json`, filtered for convergence/sampling/token/pose/seg/rate | Relevant adjacent laws: `trajectory_derived_stopping_law_v1`, `pose_null_subspace_is_ac_only_v1`, `seg_rate_breakeven_v1`, `score_marginal_lagrange_multipliers_v1`, token/Brotli/rate laws. No Markov convergence Pact score law. | No new canonical equation claimed. |
| trajectory/cap evidence | `.omx/research/ddm_tj1_20260805/*` | TJ1 already encodes `safety_bound_REPORTED`; SQ1 25/50-step rows are floor-only; cap-bound rows are not converged. | Ranked two-sided stop readback as reinforce/adopt, not new measurement. |
| terminal/free-frame pose evidence | hot state; `reports/_smoke_pose_frame0_probe.json`; `ddm_wd1`; `ddm_ph4`; `reports/ddm_bo1/posenet_pair_geometry.json` | Free-frame pose upper bound is budget-conditional; bo1 exact affine kernel says pose can be cancellable, but current vehicle needs joint staged proof; terminal-pose machinery has stale/off-chain traps. | Paper does not solve pose; it suggests tail-mass/readback hygiene only. |
| token-coder evidence | `ddm_sv2_smevr_base_rule_race_20260803.md`, `ddm_xo1_20260805/XO1_RECEIPT.md`, AR2/AR4 | `IX2TOK01` token bulk is current; SMEVR loses by +5,183 B; XO1 control ordering loses +11,561 B; match structure dominates. | Folded Markov paper for bytes unless same-coder match-structure preflight wins. |
| loss engineering evidence | `canonical_research_index_dseg_20260629.md`, `ddm_sq1_eta_seg_and_hinge_ab_20260803.md`, hot state EN1 | Margin-hinge is a measured d_seg surrogate; current hinge/tau work is wiring/A-B debt. | Classified paper as N-A for loss design. |
| sampling/verdict hygiene | `ddm_na2_negative_audit_20260803.md`, `ddm_p4x_lane_existence_birth_matrix_20260803.md`, AR2/AR4 | Prefix bias is axis-specific: pose prefixes 2.54-4.21x harder, P4X Lane n96 prefix overstates target quantity by 26.3%; selection mode silence means prefix. | Adopted `for every x` as metadata/coverage discipline, not as evidence. |

## What AR5 Did Not Measure

- No SegNet or PoseNet scorer forwards.
- No `upstream/evaluate.py`.
- No archive bytes, token bytes, candidate archive, or receiver packet.
- No n32/n120/n600 Pact measurement.
- No trainer launch, GPU/MLX job, paid dispatch, or lane claim.
- No new canonical equation.
- No protected file edit.

## Follow-On Disposition

| item | disposition | fire order |
|---|---|---|
| Two-sided stop/readback fields for adaptive iterative rows | QUEUED-WITH-FIRE-ORDER | Fire at the next SL1/SQ2/TJ1-style cached read; no scorer. |
| Terminal/free-frame pose tail-mass replay | QUEUED-WITH-FIRE-ORDER | Use existing traces only; fold if tail mass is nondecaying or nonpredictive. |
| Positive-mixture proposal bucket for candidate generators | QUEUED-WITH-FIRE-ORDER | Byte-only/support-only preflight; fold unless same-coder bytes and support improve. |
| Markov-chain theorem as token coder, pose solver, loss, or score law | FOLDED | No consumer, no S-unit derivation, no archive mechanism. |

## NEXT_IF_RESUMED

```json
{
  "schema": "ddm_ar5_next_if_resumed.v1",
  "status": "PAPER_CROSSWALK_COMPLETE_NO_SCORE",
  "paper": "arXiv:2608.03353 Markov-chain convergence via asymptotic equivalence",
  "adopt": [
    "two_sided_convergence_readback_for_adaptive_iterative_rows",
    "tail_mass_decay_hygiene_for_terminal_pose_and_free_frame_pose_reads",
    "positive_mixture_support_probe_for_candidate_generators",
    "for_every_start_or_selection_metadata_discipline"
  ],
  "already_embodied": [
    "safety_bound_REPORTED_not_converged",
    "no_rate_from_qualitative_convergence",
    "joint_seg_pose_rate_pricing_for_corrections",
    "m88_m96_prefix_and_selection_mode_hygiene"
  ],
  "refuse_as_solution_for": [
    "token_bytes_without_same_coder_match_structure_win",
    "terminal_pose_solve",
    "margin_hinge_tau_loss_wiring",
    "exact_score_or_archive_byte_claim"
  ],
  "scorer_forwards": 0,
  "evaluate_py": false,
  "archive_mutation": false,
  "pointer_moved": false,
  "follow_on_disposition": "QUEUED-WITH-FIRE-ORDER"
}
```

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
