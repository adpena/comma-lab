# Z3 Byte-Saving Archive Contract Fail-Closed - Codex 2026-05-14

parent_id_or_session: current_codex_scope_z3_byte_saving_archive_contract
inherited_directives:
- CLAUDE.md
- AGENTS.md
- .omx/research/journal_lab_grade_documentation_standard_directive_20260514.md
- .omx/research/recovery_session_20260514_directive_absolute_no_signal_loss_20260514.md
lane_id: lane_z3_balle_hyperprior_bolton_recover_20260514
status: code_and_tests_landed_locally
research_only: true
score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
gpu_spend: false

## Hypothesis

Append-only Z3HP1 packets cannot be byte-saving because they preserve all A1
bytes and add a sidecar; production code must fail closed until Z3 replaces the
A1 latent_blob inside the inner `x` payload and reconstructs latents before
HNeRV decode.

## Math

Contest score rate term is:

```text
S_rate = 25 * B / 37,545,489
```

If Z3 appends `s` bytes to the unchanged A1 inner payload of `B_a1` bytes:

```text
B_z3_append = B_a1 + s
delta_S_rate = 25 * (B_a1 + s - B_a1) / 37,545,489
             = 25 * s / 37,545,489 > 0
```

Therefore append-only Z3HP1 is a rate regression before scorer effects. It may
be a diagnostic parser/runtime artifact, but it cannot honestly carry
`byte_saving=true` or `ready_for_exact_eval_dispatch=true`.

## Citations And Cross-Refs

- Balle et al. 2018, "Variational image compression with a scale hyperprior":
  side-info must amortize against replaced entropy stream bytes.
- `.omx/research/campaign_z3_balle_hyperprior_bolton_20260514.md`: records the
  intended "replace A1 rate-y encoding" path.
- `.omx/research/z3_phase_2_council_20260514.md`: records the Balle
  amortization principle and the fallback-to-A1 safety intent.
- `.omx/research/z3_balle_hyperprior_smoke_classification_20260514_codex.md`:
  classifies prior smoke as `smoke-no-scorer`, not score evidence.

## Code Change

- `src/tac/substrates/z3_balle_hyperprior_bolton/archive.py`
  - Added `Z3CompositionArchiveContract`.
  - Added `build_composition_archive_contract()`.
  - Changed `pack_composition_archive()` to raise on non-empty append-only
    sidecars unless `allow_append_only_diagnostic=True`.
  - Append-only contracts force `byte_saving=false`, `score_claim=false`,
    `promotion_eligible=false`, and `ready_for_exact_eval_dispatch=false`.
- `experiments/train_substrate_z3_balle_hyperprior_bolton.py`
  - Emits archive-contract fields into smoke and full stats.
  - Keeps full path byte-identical-to-A1 when latent replacement is absent.
- `scripts/remote_lane_substrate_z3_balle_hyperprior_bolton.sh`
  - Completion marker is no longer hard-coded as `[contest-CUDA]`.
  - Reads stats and uses `[contest-CUDA]` only if a valid contest_cuda score
    claim exists; otherwise logs `score_claim=false`.
- `.omx/operator_authorize_recipes/substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch.yaml`
  - Set `smoke_only: true` and `smoke_validation_contract:
    training_artifact_v1` until latent replacement lands.

## Six-Hook Wire-In

- Sensitivity map: N/A; this patch is an archive-authority guard, not a scorer
  sensitivity update.
- Pareto constraint: append-only Z3HP1 is constrained to `byte_saving=false`.
- Bit allocator: no allocation change; latent replacement remains the next
  byte-closed allocator target.
- Cathedral autopilot: recipe and stats now fail closed with readiness false.
- Continual learning: no empirical anchor appended; no dispatch occurred.
- Probe disambiguator: not needed; the append-only inequality above is decisive.

## Verification

```bash
PYTHONPATH=src:upstream .venv/bin/python -m pytest \
  src/tac/substrates/z3_balle_hyperprior_bolton/tests/test_z3_substrate.py -q
# 38 passed in 1.17s

bash -n scripts/remote_lane_substrate_z3_balle_hyperprior_bolton.sh
bash -n scripts/operator_authorize_substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch.sh

PYTHONPATH=src:upstream .venv/bin/python -m py_compile \
  src/tac/substrates/z3_balle_hyperprior_bolton/archive.py \
  src/tac/substrates/z3_balle_hyperprior_bolton/inflate.py \
  src/tac/substrates/z3_balle_hyperprior_bolton/__init__.py \
  experiments/train_substrate_z3_balle_hyperprior_bolton.py

.venv/bin/python tools/run_modal_smoke_before_full.py \
  --recipe substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch --dry-run
# smoke_validation_contract=training_artifact_v1
# recipe declares smoke_only: true
# would stop after SMOKE because smoke-only is set

git diff --check -- <touched Z3 files>
```

## Provenance

| Element | Value | Verification |
|---|---|---|
| HEAD before local patch | `4c94e09f0d92c42bd3faf7f6997229cf457652f3` | `git rev-parse HEAD` |
| Code patch time | `2026-05-14T16:38:02Z` | `date -u +%FT%TZ` |
| Dispatch | none | operator scope: do not dispatch |
| Score axis | none | no auth eval run |
| Score claim | false | stats/tests enforce false authority |

## Stop/Continue Thresholds

- Stop exact eval dispatch if `archive_contract.layout` is
  `append_only_z3hp1_diagnostic` or `a1_byte_identical_fallback`.
- Continue to exact eval only after `layout=latent_replacement_z3hp1` (or its
  successor) replaces A1 latent bytes, reconstructs latents before decode, and
  emits `byte_saving=true` from measured archive bytes.

## Reactivation Criteria

Reopen Z3 for score-bearing dispatch when a byte-closed latent-replacement
format lands with:

1. Inner `x` payload no longer contains A1 `latent_blob`.
2. Inflate reconstructs latents from Z3-coded bytes before HNeRV decode.
3. Full-frame inflate parity or exact same-runtime auth-eval packet is reviewed.
4. Archive bytes are strictly lower than the A1 source archive.

## Operator-Routable Decisions

1. Keep Z3 recipe smoke-only now: $0 until operator dispatches a research smoke;
   lowest risk because it prevents no-scorer smoke from green-lighting full.
2. Next implementation path: build the latent-replacement inner `x` grammar;
   medium risk because runtime parse/decode must split decoder state from
   latents without relying on A1's old latent_blob.
3. Do not retire Z3: this patch classifies only append-only packets as
   non-byte-saving; Balle hyperprior remains valid if it replaces bytes.

## Resume Instructions

Next agent should implement `latent_replacement_z3hp1` as a new typed layout,
update generated inflate.py to parse decoder state without decoding A1
latent_blob when Z3 latents are present, and add full-frame inflate parity
before any exact-eval dispatch.
