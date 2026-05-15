# PR101 FEC6 0.192 Axis Check + PR106 Review Join - 2026-05-15

## PR101 FEC6 0.192 Classification

The `0.1920513168811056` result is a legitimate `[contest-CPU]` score for the
PR101 FEC6 fixed-Huffman K16 selector packet:

- archive: `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip`
- archive sha256: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- bytes: `178517`
- `[contest-CPU]` result: `experiments/results/modal_auth_eval_cpu/archive_6bae0201fb08/modal_cpu_auth_eval_result.json`
- `[contest-CUDA/T4]` paired result: `0.22621002169349796`
- CUDA result-review packet: `.omx/research/pr101_fec6_fixed_huffman_k16_cuda_result_review_20260515_codex.json`

Conclusion: this is a real CPU-axis improvement and a real small CUDA
improvement over the FEC3/K8 CUDA baseline (`0.22626723761043824`), but it is
not a CUDA frontier or leaderboard-breaker. It must not be promoted as a
`<0.192` contest-CUDA result.

## Film-Grain / Selector / Water-Fill Status

Current same-frame PR101 selector polishing is close to exhausted on the byte
axis:

- PR101 FEC6 byte-only gap to `<0.192` on `[contest-CPU]`: about `78` bytes.
- Realistic same-frame byte-saving bound from the FEC6 byte-escape profile:
  about `16` bytes.
- Existing PR101 FEC3/K8 CPU near-miss remains `0.19209788683213053`; FEC6/K16
  improves CPU to `0.1920513168811056` but does not cross `<0.192`.

The broader film-grain/selector family is not mathematically exhausted. The
measured conclusion is narrower: the current proxy-ranked CPU/MPS selector and
HDM8 film-grain water-fill variants did not transfer into a frontier-improving
CUDA packet. Future work should be CUDA-in-loop or use a paired CPU/CUDA
component model before spending another exact eval.

## PR106 Recode Exact-Review Join

Landed code hardening in `tools/profile_pr106_latent_sidecar_recode.py`:

- indexes `.omx/research/*result_review*.json` packets with
  `schema=tac_result_review_packet_v1`;
- joins exact-CUDA review evidence by emitted candidate archive SHA-256;
- removes stale `exact_cuda_auth_eval_missing` when a matching exact review
  already exists;
- marks the row with `exact_cuda_result_review_already_exists` and keeps
  `score_claim=false`, `promotion_eligible=false`, and
  `ready_for_exact_eval_dispatch=false`.

Focused regression:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest -p no:cacheprovider -q src/tac/tests/test_pr106_latent_sidecar_recode.py
```

Result: `12 passed`.

Regenerated PR106 profile:

- `experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/profile.with_proofs.json`
- `experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/profile.with_proofs.md`

Format `0x0A` now links to
`.omx/research/pr106_hdm10_hlm3_format0a_exact_cuda_result_review_20260515_codex.json`
with exact `[contest-CUDA]` score `0.2063310355127786`; it is blocked as
already reviewed rather than as missing exact CUDA eval.

## D4 Smoke Harvest

D4 smoke harvest completed:

- lane: `lane_d4_wyner_ziv_frame_0_substrate_20260514`
- call id: `fc-01KRN4AAM138RHZT6WXKXK464B`
- label: `substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch_20260515T061339Z__smoke__50ep`
- elapsed: `956.2302869849999` seconds
- estimated cost: `$0.15671551925587496`
- status: `completed_modal_training_recovered_no_score_claim`
- output artifact root: `experiments/results/lane_substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch_20260515T061339Z__smoke__50ep_modal`

This was a 200-pair truncated smoke and intentionally skipped auth eval:
`stage_4_truncated_pair_smoke_skips_auth_eval max_pairs=200 required_pairs=600`.
The run emitted a WZF01 archive of `657455` bytes, with best validation
Lagrangian at epoch `0`, so it is not promotable and makes no score claim.

Engineering follow-up before another D4 run:

- set `CUBLAS_WORKSPACE_CONFIG` in the remote trainer environment to reduce
  deterministic-warning noise;
- do not interpret this smoke as a score result;
- only launch a full 600-pair auth-eval-bearing run if the next recipe reduces
  archive proxy bytes or validation Lagrangian enough to justify spend.
