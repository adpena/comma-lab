---
name: Lane MDL/Bayesian (MacKay) — Level 0 → Level 1 SCAFFOLD landed
description: 2026-04-30. Phase 3 Lane MDL/Bayesian (codec-selection framework over OTHER codec families) advanced from Level 0 (sketch only) to Level 1 (SCAFFOLD). 27/27 synthetic tests passing. Council design doc landed. impl_complete gate satisfied. Remaining 6 gates (real-archive empirical / contest-CUDA / STRICT preflight / 3-clean-pass review / memory entry / deploy runbook) deferred to Phase 2 — this lane is a META framework that consumes other codecs' results.
type: project
authoritative_for: lane_mdl_bayesian_level1_scaffold
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## TL;DR

Lane MDL/Bayesian is **NOT a new codec** — it is a Bayesian codec-selection framework (MacKay-channeled) that consumes per-codec L_total accountings from sibling lanes (Lane Ω-W-V2, Lane J-NWC, Lane 20 Ballé, Lane SH static, etc.) and produces:
- Posterior weight ranking (softmax over -L_total + log_priors)
- Bayes factor pairwise comparisons (log2 odds)
- Laplace-approximation log evidence (for advanced model comparison)
- Occam's razor pre-flight check that **refuses to ship codecs whose L(M) exceeds achievable y-stream savings** (Quantizr's adversarial objection operationalized)
- Bayesian model averaging (rare; usually one codec dominates)

**Direct byte savings: 0 ± 200 B** (the framework ships nothing in archive.zip).
**Indirect score lift via better stack selection: -0.005 to -0.015 [prediction]**.

## Files landed

- `src/tac/mdl_bayesian_codec.py` — 366 LOC; required-keyword args; no silent defaults; pure CPU; module-level `# ROUNDTRIP_NOT_REQUIRED:` waiver justified (it's a meta-framework, not a byte codec)
- `src/tac/tests/test_mdl_bayesian_codec.py` — 27 tests; covers primitives (mdl_total_bits, bayes_factor_log2, laplace_log_evidence), MDLCodecResult dataclass, ranking under softmax-underflow, BMA, OccamCheck (passing + rejecting), prior derivation
- `.omx/research/council_lane_mdl_bayesian_design_20260430.md` — full council deliberation; MacKay LEAD + Shannon LEAD seats; Hinton (variational MDL co-author 1993) channeled; Schmidhuber + Quantizr + Hotz + Selfcomp seats; Hotz seat reduced scope to ~250 LOC pragmatic implementation

## Council verdict

**Adopt: Implement Lane MDL as a thin codec-comparison framework, not a new codec.** All 7 council voices signed off (Quantizr YELLOW on direct savings — accurate; framework is informational not byte-saving). Hotz pushed for tiny audit-friendly module.

## Subagent-induced tweak

Linter added a module-level `# ROUNDTRIP_NOT_REQUIRED:` comment justifying that this `*_codec.py` filename does NOT have an encode/decode roundtrip pair — that's correct behavior for a meta-framework.

## Tests

```
PYTHONPATH=src .venv/bin/python -m pytest src/tac/tests/test_mdl_bayesian_codec.py
27 passed in 0.13s
```

Notable test: `test_rank_codecs_smallest_l_total_wins` initially failed because softmax with L_total differences > 100 bits underflows to 0 weights → unstable argsort. Fixed by adding composite-key tie-breaking on (-weight, total_bits) so that even when softmax degenerates, the ranking is monotonic in -L_total.

## Lane registry status

```
lane_mdl_bayesian: level 0 → 1
gates: impl_complete=true; remaining 6 gates false
```

## Predicted band

[prediction] -0.005 to -0.015 score (indirect via better stack selection); 0 ± 200 B direct savings.

## Phase ordering ahead

- Phase B (Level 1) ✅ THIS COMMIT
- Phase C (Level 2 prep) — wire to Lane Ω-W-V2 + Lane J-NWC + Lane 20 Ballé result JSONs
- Phase D (Level 2) — produce ranking report on actual artifacts; cross-check vs contest-CUDA
- Phase E (Level 3 path) — STRICT preflight check (any new codec lane MUST report L_total + ship MDL-ranked winner)
- Phase F — 3-clean-pass adversarial review

## Cross-references

- CLAUDE.md "Council conduct — non-negotiable"
- CLAUDE.md "MacKay's specific contributions" (this lane is direct expression of his role)
- `feedback_production_hardened_standard_definition_20260430.md` (Level 3 bar)
- `.omx/research/council_lane_mdl_bayesian_design_20260430.md` (the full council design)
- `project_phases_2_3_4_design_implementation_math_provenance_20260429.md` §"Lane 16 Bayesian MDL"
- `src/tac/balle_hyperprior_renderer.py` (Lane 20; one of the codecs MDL ranks)
- `src/tac/arithmetic_qint_codec.py` (Lane SH; the static-prior baseline)
- MacKay 2003 ITILA Ch. 28; Rissanen 1978; Hinton & van Camp 1993
