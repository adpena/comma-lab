# V6 = Proof-Carrying Evaluator-Equivalent Program Compiler — definition + approval checklist

UTC 2026-06-09 · operator grand-council verdict: **V6 design target APPROVED; V6 implementation NOT yet
approved** (gated on the checklist below). This memo resolves the V6 reservation with the operator's
definition and binds the approval gates.

## What V6 IS (the operator's definition)
V6 is NOT a new carrier. V6 is the **synthesis layer** — the evidence-lattice + experiment-planning
compiler that unions V1–V5 under one proof law and emits the shortest executable witness `evaluate.py`
accepts. The atlas/atom lane is `Vatlas` (a proposal engine), distinct from V6.

```
V0  source_recode (fp11 brotli)      — the current CPU frontier anchor (0.19199)
V1  HiNeRV / HNeRV dense carrier
V2  SNeRV source-state carrier
V3  evaluator-action compiler / waterfiller   (the exact-ΔS judge)
V4  PACT-NeRV-VQ composed-latent / codebook carrier
V5  PR110++ selector / menu action system
Vatlas  evaluator-atlas atoms (inverse-steg + cooperative-receiver miners)
V6  = PROOF-CARRYING COMPILER = union(V1..V5, Vatlas) + Evidence Constitution + Dual-Optimization
      + experiment planner, all judged by V3 under exact ΔS.
```

## The compiler's objective (the only law)
```
min over A in L, T(A) <= 30 min:
    100*d_seg(X, I(A)) + sqrt(10*d_pose(X, I(A))) + 25*|A|/37,545,489
```
A=archive program · I(A)=inflated frames · X=source · |A|=archive.zip bytes · T=eval runtime.
Every carrier/codec/selector/codebook/source-state/atom/checkpoint → a typed `CandidateActionEvaluation`
(base+candidate sha, d_seg, d_pose, bytes, ΔS, authority_tier, metric_family, stale, eligibility).
Admission: **ΔS < 0**, no exceptions. Score-roadmap moves only on a contest-axis `exact_evaluate` row;
promotion only on paired CPU+CUDA on the same archive_sha256.

## V6 DESIGN approval — SATISFIED (committed this session)
- [x] Evidence Constitution (`pact_evidence_constitution_20260609.md`) — authority_tier × metric_family × eligibility.
- [x] Compiler Dashboard generator (`tools/render_pact_compiler_dashboard.py`) — living vehicle index.
- [x] Dual-Optimization Principle (`dual_optimization_principle_..._20260609.md`) — intrinsic × contextual.
- [x] Vehicle registry (`tac.optimization.composition_carrier_registry`).
- [x] `CandidateActionEvaluation` canonical (`tac.optimization.harvest_evidence`) + metric-family firewall + authority-tier eligibility (firewall-tested, 18 tests).
- [x] North-star + thousand-year framing (`north_star_..._20260609.md`).
- [x] Codex reinforcement packet (`codex_reinforcement_packet_20260609.md`).

## V6 IMPLEMENTATION approval — NOT yet (the 10-item gate)
1. [ ] ep1000 clean-PR95 exact eval ingested through V3 (RUNNING — the next authority datum).
2. [ ] artifact indexer built (`tools/index_pact_artifacts.py`) — THIS turn.
3. [ ] historical artifacts backfilled into typed rows.
4. [ ] each active vehicle has a `CandidateActionEvaluation` path.
5. [ ] PR110++ replay produces action rows.
6. [ ] PACT-VQ maturity audit completed (DONE — verdict HiNeRV-retrofit; needs one exact row).
7. [ ] SNeRV source-forward produces exact rows OR is explicitly blocked.
8. [ ] V3 materializer proposes + exact-evaluates >= 1 atom.
9. [ ] commutator planner has >= 1 measured pair.
10. [ ] dashboard regenerates all of the above automatically.

## Canonical exemplars (V6 must compare every lane against these)
- **PR95 = dense-carrier exemplar:** 229K HNeRV decoder, 28-d pair latents, INT8+Brotli, 8-stage
  curriculum (CE-Seg → τ-softplus → smooth → QAT → hard-pixel/C1a → λ → σ → Muon), ~178 KB, ~0.20.
- **PR110 = action-menu exemplar:** mode catalog × per-pair selector stream + Huffman indices + exact
  ΔS; exploits frame0→PoseNet / SegNet-blind asymmetry. (`hnerv_fec6_fixed_huffman_k16`.)
  Reference specs queued: `reference_pr95_dense_carrier_spec.md`, `reference_pr110_selector_menu_spec.md`.

## The 10 adversarial bugs (each "yes" is a defect V6 must make impossible)
telemetry looks like exact · advisory updates score roadmap · stale base reused · SNeRV proof without
exact score · PACT-VQ stays vague · PR110++ waits behind neural lanes · sidecar enters without pays_rent
· uniform codec without evidence · dashboard goes stale · subagent works without a vehicle assignment.

## Discipline (binding)
Heavy slot stays on clean PR95 (one Metal slot). Light slots: indexer, specs, Codex prep, dashboard
refresh. No branch before ep1000. No proxy/advisory roadmap update. Every lane in V3 currency. Never
auto-kill. No upstream edits for authority.
