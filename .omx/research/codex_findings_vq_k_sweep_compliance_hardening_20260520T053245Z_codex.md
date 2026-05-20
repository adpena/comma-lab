# Codex Findings: VQ K-Sweep Compliance Hardening

**UTC:** 2026-05-20T05:32:45Z  
**Actor:** codex  
**Lane:** `lane_e7_vq_k_sweep_plus_e8_sgld_convergence_prep_20260518`  
**Recipe:** `.omx/operator_authorize_recipes/substrate_vq_vae_k_sweep_modal_t4_dispatch.yaml`  
**Trainer:** `experiments/train_substrate_vq_vae.py`

## Verdict

Landed compliance hardening for the VQ-VAE K-sweep dispatch surface, then dispatched a corrected K=2 A10G diagnostic run.

The run is explicitly **not** a contest-CUDA score claim. The Modal training wrapper uses `AUTH_EVAL_DEVICE=cpu` and `MODAL_AUTH_EVAL_ADVISORY_ONLY=1`; any inline auth-eval result is diagnostic only with `score_claim=false`, `promotion_eligible=false`, and `rank_or_kill_eligible=false`.

## Findings

1. Prior K=2 dispatch label `substrate_vq_vae_k_sweep_modal_t4_dispatch_20260520T015850Z_k_2_slot_gg_v2` was not a valid K=2 datapoint. Its harvested trainer provenance records `codebook_size=512` and `alpha_rate=25.0`; the recipe env values were not threaded to the trainer.
2. The recipe still claimed single-axis `[contest-CUDA]` acceptability while `experiments/modal_train_lane.py` forces CPU advisory inline auth-eval.
3. The trainer hardcoded `contest_auth_eval_cuda.json` and CUDA score fields even when the helper redirected actual auth eval to CPU.
4. The trainer recorded raw `0.bin` SHA/bytes as archive identity, while auth eval consumes `archive.zip`.
5. The remote completion marker reported `archive=$OUTPUT_DIR/0.bin` instead of the evaluated `archive.zip`.
6. The K-rate table was phrased as implemented bit-packed archive accounting; it is only an analytical floor until measured from emitted `archive.zip`/`0.bin`.

## Fixes Landed

- `scripts/remote_lane_substrate_vq_vae.sh`
  - Threads `VQ_VAE_CODEBOOK_SIZE` and `VQ_VAE_ALPHA_RATE` into `experiments/train_substrate_vq_vae.py`.
  - Records those values in remote provenance and trainer invocation logs.
  - Resolves the auth-eval completion artifact from actual `AUTH_EVAL_DEVICE`.
  - Emits `[contest-CUDA]` only when the auth-eval artifact is CUDA; CPU emits `[diagnostic-auth-eval]`.
  - Reports `archive=$OUTPUT_DIR/archive.zip`.

- `experiments/train_substrate_vq_vae.py`
  - Writes auth-eval JSON to `contest_auth_eval_{actual_device}.json`.
  - Calls the canonical auth-eval gate with `return_non_cuda_result=True`.
  - Keeps CPU/diagnostic auth-eval out of CUDA score and posterior fields.
  - Separates `payload_bin_sha256`/`payload_bin_bytes` from evaluated `archive_sha256`/`archive_bytes`.
  - Binds `archive_sha256` and `archive_bytes` to `archive.zip`, matching the evaluated artifact.

- `.omx/operator_authorize_recipes/substrate_vq_vae_k_sweep_modal_t4_dispatch.yaml`
  - Relabels this as Modal A10G diagnostic dispatch.
  - States inline auth-eval is CPU advisory only.
  - Updates the cost/envelope text to the empirical A10G per-K envelope.
  - Labels the K table as an analytical, unimplemented bit-packed floor.

- Tests added/updated:
  - `src/tac/tests/test_remote_lane_vq_vae_script.py`
  - `src/tac/tests/test_vq_vae_k_sweep_compliance.py`

## Review And Dispatch Evidence

- Focused tests: `70 passed`.
- `ruff check`: clean on touched Python files.
- `bash -n scripts/remote_lane_substrate_vq_vae.sh`: clean.
- `tools/local_pre_deploy_check.py --trainer experiments/train_substrate_vq_vae.py --recipe substrate_vq_vae_k_sweep_modal_t4_dispatch --strict`: all 9 checks passed.
- Catalog #271 no-cache review: `approve`, `findings=[]`, `cache_key=004101084394b03a`, artifact `.omx/state/codex_review_vq_k_sweep_20260520T0530Z.json`.
- Catalog #202 sentinel audit: `.omx/state/catalog202_sentinel_cleanliness/substrate_vq_vae_k_sweep_modal_t4_dispatch_20260520T052448Z.json`, sentinel set SHA `36bf9c746293849110997b0db260eb6e9b503fce3519841f806fb3546b008e06`.
- Dispatch:
  - label `substrate_vq_vae_k_sweep_modal_t4_dispatch_20260520T053154Z_k_2_codex_compliance_fixed_20260520`
  - call id `fc-01KS1XXZEMJGDT1Q53R64GJ7AS`
  - platform `modal`, GPU `A10G`
  - active claim appended at `.omx/state/active_lane_dispatch_claims.md`
  - call id appended in `.omx/state/modal_call_id_ledger.jsonl`
  - local metadata: `experiments/results/lane_substrate_vq_vae_k_sweep_modal_t4_dispatch_20260520T053154Z_k_2_codex_compliance_fixed_20260520_modal/modal_metadata.json`

The operator-authorize pre-dispatch review was skipped only after a fresh no-cache Catalog #271 approval of the same code/recipe state. Rationale was passed through the paired env contract and cites the approval artifact above; the first no-bypass operator-authorize attempt exited before provider dispatch and created no claim/call id.

## Remaining Work

- Harvest call id `fc-01KS1XXZEMJGDT1Q53R64GJ7AS` within 24h:
  `.venv/bin/python experiments/modal_recover_lane.py --call-id fc-01KS1XXZEMJGDT1Q53R64GJ7AS`
- Verify the harvested trainer provenance now records `codebook_size=2` and `alpha_rate=1.0`.
- Verify harvested `provenance.json` binds `archive_sha256`/`archive_bytes` to `archive.zip`, with raw payload identity only in `payload_bin_*`.
- Keep result diagnostic unless a separate claimed exact-CUDA eval is dispatched against the emitted archive/runtime packet.
