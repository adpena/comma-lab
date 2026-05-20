<!-- Catalog #344 canonical equation cross-ref: Percepta integration plan empirical anchors align with `tac.canonical_equations` registry per `percepta_programs_in_weights_pact_integration_v1` candidate registration upon empirical anchor landing. Per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — no mutation. -->

# Codex Findings: Percepta Programs-In-Weights Pact Integration Plan

UTC: 2026-05-20T18:55:19Z  
Owner: Codex  
Lane: `lane_percepta_microprogram_pact_integration_20260520`  
Status: `research_only=true`; planning/helper artifact only; no score claim.

## Scope

Task: translate the Percepta/WebAssembly-in-weights idea into a contest-faithful,
minimal, testable Pact artifact plan for PR110/FEC6 constraints.

Non-negotiables applied:

- PR110 live submission files were not edited.
- Inflate must remain deterministic, scorer-free, and network-free.
- Archive bytes remain the score-bearing budget; source-embedded payloads are a
  compliance risk.
- Exact eval custody is required before any promotion or score/rank claim.
- A tiny circuit must beat simpler q-symbol or byte-only edits after rate cost.
- Full WASM interpreter shipment is a no-go for this lane.

## Verdict

The contest-faithful translation is **not** "ship WASM." It is:

1. **First prototype surface: weight-embedded tiny circuit.** Compile a 1-3 op
   correction rule into existing FEC6 decoder q-symbol choices, with zero
   runtime patch and no new archive section. This is safest because it avoids an
   uncharged source-side payload and forces all data into charged/consumed model
   bytes.
2. **Second prototype surface: decoder-side microprogram.** Only if the
   weight-embedded route finds a measured response but needs a branch/table that
   cannot be represented by q edits. Cap the bytecode at a few dozen bytes and
   the runtime patch as generic, data-free interpreter logic.
3. **No-go surface: general WASM interpreter.** Too many bytes, too much runtime
   complexity, and too easy to become an uncharged side channel. Percepta is
   useful as a plausibility proof for compiled deterministic circuits, not as
   score evidence or a runtime design to copy.

## Artifact Added

Small planning helper:

- `src/tac/optimization/percepta_microprogram_plan.py`
- `tools/plan_percepta_microprogram_candidate.py`
- `src/tac/tests/test_percepta_microprogram_plan.py`

The helper emits a `percepta_microprogram_plan.v1` JSON object. It computes:

- signed archive-byte rate delta via `25 * byte_delta / 37_545_489`;
- runtime patch bytes as a review/compliance gate, not as a score-rate term;
- a simple-edit hurdle using the known same-runtime polish floor of 78 bytes
  unless a measured best q-edit delta is provided;
- prototype blockers for full interpreters, unbounded opcodes, scorer/runtime
  violations, PR110 live-file touches, excessive ops, and excessive bytecode;
- promotion blockers for missing exact-eval custody or failure to beat q/byte
  edits;
- cheapest empirical smoke commands using existing Pact tools and verified CLI
  flags.

Default helper output is intentionally:

`PROTOTYPE_GO_PROMOTION_BLOCKED`

because the default two-op weight-embedded smoke is contest-faithful enough to
materialize, but has no measured component gain and no exact-eval custody.

## Go / No-Go Gates

Prototype GO requires all of:

- `surface != general_wasm_interpreter`
- opcodes are in the bounded tiny set:
  `select_masked`, `add_i8_saturating`, `clamp_u8`, `affine_i8`,
  `mul_pow2`, `lookup_const4`, `branch_on_selector_bit`, `xor_selector_bit`,
  `const_i8`
- operation count <= 16
- encoded program bytes <= 64
- runtime patch bytes <= 512
- deterministic inflate
- no scorer imports or calls during inflate
- no network or external I/O during inflate
- no edits to PR110 live submission files

Promotion GO additionally requires all of:

- candidate archive SHA-256 recorded;
- runtime tree SHA-256 recorded;
- inflated output manifest SHA-256 recorded;
- terminal dispatch claim recorded;
- axis tag is `[contest-CPU]` or `[contest-CUDA]`;
- projected total score delta, including rate, beats:
  `min(best_simple_q_edit_delta_score, -25 * 78 / 37_545_489)`;
- raw inflate control proves the candidate is not a no-op and only changes the
  intended output surface;
- same-runtime exact eval exists before any score/rank/promotion claim.

No-go conditions:

- full WASM interpreter or general VM in the runtime;
- source-embedded data payload masquerading as code;
- byte increase without component delta that beats q/byte alternatives;
- missing exact-eval custody for promotion;
- any scorer, network, nondeterminism, or PR110 live-file touch.

## Cheapest Empirical Smoke

The cheapest smoke is zero-runtime and q-domain first:

```bash
.venv/bin/python tools/plan_percepta_microprogram_candidate.py \
  --prototype-id percepta_weight_embedded_rgb_bias_gate_smoke \
  --output experiments/results/percepta_microprogram_smoke/plan.json

.venv/bin/python tools/probe_op3v3_decoder_mutation_feasibility.py \
  --archive-bin experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip \
  --max-offsets-per-tensor 4 \
  --deltas -1,1 \
  --output experiments/results/percepta_microprogram_smoke/percepta_weight_embedded_rgb_bias_gate_smoke_feasibility.json

.venv/bin/python tools/materialize_decoder_q_candidates.py \
  --archive-bin experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip \
  --feasibility experiments/results/percepta_microprogram_smoke/percepta_weight_embedded_rgb_bias_gate_smoke_feasibility.json \
  --limit 2 \
  --output-dir experiments/results/percepta_microprogram_smoke/percepta_weight_embedded_rgb_bias_gate_smoke_archives \
  --manifest-output experiments/results/percepta_microprogram_smoke/percepta_weight_embedded_rgb_bias_gate_smoke_manifest.json

.venv/bin/python tools/run_decoder_q_candidate_inflate_controls.py \
  --runtime-dir experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir \
  --candidate-root experiments/results/percepta_microprogram_smoke/percepta_weight_embedded_rgb_bias_gate_smoke_archives \
  --output-root experiments/results/percepta_microprogram_smoke/percepta_weight_embedded_rgb_bias_gate_smoke_inflate \
  --output experiments/results/percepta_microprogram_smoke/percepta_weight_embedded_rgb_bias_gate_smoke_inflate_controls.json

.venv/bin/python tools/run_decoder_q_candidate_advisory_batch.py \
  --runtime-dir experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir \
  --candidate-root experiments/results/percepta_microprogram_smoke/percepta_weight_embedded_rgb_bias_gate_smoke_archives \
  --baseline-raw experiments/results/percepta_microprogram_smoke/percepta_weight_embedded_rgb_bias_gate_smoke_inflate/baseline_raw \
  --output-root experiments/results/percepta_microprogram_smoke/percepta_weight_embedded_rgb_bias_gate_smoke_advisory \
  --axis-label '[macOS-CPU advisory]' \
  --max-candidates 2 \
  --output experiments/results/percepta_microprogram_smoke/percepta_weight_embedded_rgb_bias_gate_smoke_advisory_summary.json
```

This smoke stays non-promotional. It can only answer whether a tiny compiled
q-circuit has raw-output visibility and advisory component response. Any
positive result must still go through byte-closed candidate packaging and exact
eval custody on the matching contest axis.

## Compliance Risks

- **Uncharged payload risk:** if the "program" lives in source code rather than
  archive bytes or existing weights, it may become a side channel. Keep source
  code generic; data-bearing program bits must be charged or embedded into
  already charged model bytes.
- **False authority risk:** Percepta supports feasibility of program-as-weights,
  not a Pact score claim. Every Pact claim must come from local raw inflate and
  exact eval artifacts.
- **Runtime complexity risk:** even a tiny decoder microprogram can exceed its
  benefit if runtime bytes, review burden, or nondeterministic behavior rise.
- **Axis drift risk:** `[macOS-CPU advisory]` can rank candidates only. Promotion
  requires same archive/runtime on `[contest-CPU]` or `[contest-CUDA]`.
- **PR110 freeze risk:** all experiments must occur in new candidate artifacts
  under `experiments/results/`; the live PR110 source tree stays untouched.

## Canonical-vs-Unique Decision Per Layer

- Canonical: FEC6 decoder-q mutation helpers, existing q-candidate materializer,
  raw inflate controls, and advisory batch runner. These already encode the
  byte-closed/no-op discipline this idea needs.
- Canonical: lane registry and exact-eval custody concepts. This prevents a
  research-only helper from being mistaken for a score-bearing candidate.
- Unique: `percepta_microprogram_plan.v1` gate model. This is specific to the
  programs-in-weights question because it distinguishes weight-embedded
  circuits, decoder-side microprograms, and full interpreters.
- Unique: tiny-opcode whitelist and q/byte-edit hurdle. These make the Percepta
  idea compete against the actual cheaper alternatives in FEC6.

## Six-Hook Wire-In

1. Sensitivity map: deferred; first smoke consumes existing OP3/FEC6 q-target
   surfaces rather than emitting new sensitivity anchors.
2. Pareto constraint: active through the rate-adjusted q/byte-edit hurdle.
3. Bit allocator: active by forcing comparison against simpler q-symbol or
   byte-only edits.
4. Cathedral/autopilot dispatch: blocked until `PROMOTION_GO`; helper output is
   planning-only.
5. Continual-learning posterior: no empirical anchor yet; update only after a
   measured candidate.
6. Probe-disambiguator: active by testing weight-embedded circuit first and
   rejecting full interpreter by construction.

## Verification Run

Focused verification performed:

```bash
.venv/bin/python -m pytest src/tac/tests/test_percepta_microprogram_plan.py -q
# 9 passed

.venv/bin/python tools/plan_percepta_microprogram_candidate.py \
  --prototype-id percepta_cli_smoke \
  --runtime-patch-bytes 511 \
  --output /tmp/percepta_microprogram_plan_smoke.json

.venv/bin/python tools/lane_maturity.py validate
# OK - 1050 lane(s) validated cleanly.
```

## Next Concrete Action

Run the cheapest q-domain smoke above. Continue only if raw inflate controls
prove non-no-op consumption and advisory response beats the planner hurdle. If
the first two q candidates do not clear that bar, do not escalate to a
decoder-side microprogram; stay with simpler q/byte edits or component-moving
Rule #6 paths.
