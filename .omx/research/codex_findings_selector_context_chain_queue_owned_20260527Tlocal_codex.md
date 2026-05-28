# Codex Findings - Selector Context Chain Queue Ownership

Date: 2026-05-27

## What Landed

- Generalized `selector_stream_context_recode_v1` from FECa scale/alpha tuning into a selector-codec family materializer.
- Added FEC8 Markov/order families to the same materializer sweep:
  - `fec8_markov_static_order1`
  - `fec8_markov_adaptive_order1`
  - `fec8_markov_static_order2`
- Added chain metadata so the P11 selector codec is explicitly composed after upstream P19/P18 scorer-region repair and before downstream P15 repack.
- Wired queue contexts through `experiment_queue.v1` so codec families, upstream entropy positions, downstream materializer targets, chain parent artifacts, and full-frame parity proof paths are not lost at execution time.
- Hardened `json_completion_contract` so a runtime-consumption proof can be accompanied by a shell full-frame parity proof without forcing the shell proof to carry runtime-adapter identity fields it does not own.

## Queue-Owned Artifact

Queue artifacts:

- `.omx/research/selector_context_chain_queue_owned_20260527Tlocal/selector_context_chain_backlog.json`
- `.omx/research/selector_context_chain_queue_owned_20260527Tlocal/selector_context_chain_contexts.json`
- `.omx/research/selector_context_chain_queue_owned_20260527Tlocal/selector_context_chain_work_queue.json`
- `.omx/research/selector_context_chain_queue_owned_20260527Tlocal/selector_context_chain_execution_queue.json`

Executed output:

- `/Volumes/VertigoDataTier/experiments/results/selector_context_family_chain_queue_owned_20260527Tlocal/feca_selector_reparameterization_manifest.json`
- `/Volumes/VertigoDataTier/experiments/results/selector_context_family_chain_queue_owned_20260527Tlocal/submission_dir/archive.zip`

Queue status:

- `selector_context_chain_queue_owned_20260527Tlocal`: 1/1 step succeeded after the postcondition contract hardening.

## Empirical Result

Best remains FECa adaptive blend:

- selected codec: `fec10_adaptive_blend`
- selected scale/alpha: `64 / 1`
- selector payload: `236 -> 220` bytes
- archive bytes: `178546 -> 178530`
- archive SHA-256: `18e3155fbbbe9ab23e1c21bc0d99ba8d18657a71c3129fc5ff9e0405b67d1669`
- full-frame parity proof: reused and verified against the candidate archive

Markov/order family sweep result:

- `fec8_markov_static_order2`: 239 bytes, `-3` saved bytes versus source FECa payload
- `fec8_markov_static_order1`: 245 bytes, `-9` saved bytes
- `fec8_markov_adaptive_order1`: 270 bytes, `-34` saved bytes

Interpretation: on the live PR110 selector stream, the FECa scale64/alpha1 blend is still the P11 local optimum among the tested integer-codeword selector families. The important landing is that this is now a queue-owned family sweep result and not an ad hoc conclusion.

## PR95/HNeRV MLX Control Arm

Ran native MLX one-step timing smokes for the control arm:

- Stage 1: `.omx/research/pr95_mlx_control_arm_stage1_20260527Tlocal/run_summary.json`, `0.0441` seconds/step
- Stage 5: `.omx/research/pr95_mlx_control_arm_stage5_20260527Tlocal/run_summary.json`, `0.0312` seconds/step
- Stage 8: `.omx/research/pr95_mlx_control_arm_stage8_20260527Tlocal/run_summary.json`, `0.0303` seconds/step

All three emitted byte-closed smoke archives and false-authority manifests. These are training-control heartbeat artifacts only, not score or promotion authority.

## Next Engineering Move

The next selector/context step is not more P11-only polishing. The queue should now consume P18/P19 component-response artifacts and emit concrete upstream repair materializers, then run this P11 family recode and P15 repack as a downstream chain. The next PR95 step is to run the existing public-archive export + PyTorch export parity path on stage 8 with source-video targets, then promote that into the local training queue as the first real PR95/HNeRV MLX reproduction control arm.
