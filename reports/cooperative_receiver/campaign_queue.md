# Cooperative-Receiver Campaign Queue

- schema: `tac_cooperative_receiver_campaign_queue_v1`
- planning_only: `true`
- score_claim: `false`
- ready_for_exact_eval_dispatch: `false`
- meta_insight: Do not spend archive bytes on information the fixed scorer/decoder can infer from shared knowledge; every row remains planning-only until byte-closed exact eval exists.

| rank | campaign | predicted delta | cost | smoke | promotion gate |
|---:|---|---:|---:|---|---|
| 1 | `darts_confirmed_time_traveler_config` | `[-0.06, -0.045]` | `[3.0, 8.0]` | `PYTHONPATH=src:upstream:$PWD .venv/bin/python tools/smoke_time_traveler_l5_autonomy_macos_cpu.py --help` | DARTS config materialized + TT5L archive roundtrip + paired [contest-CUDA]/[contest-CPU] exact eval |
| 2 | `time_traveler_world_model_substrate` | `[-0.043, -0.023]` | `[3.0, 8.0]` | `PYTHONPATH=src:upstream:$PWD .venv/bin/python tools/smoke_time_traveler_l5_autonomy_macos_cpu.py --help` | TT5L archive roundtrip + macOS advisory smoke + paired exact eval |
| 3 | `sabor_boundary_only_renderer` | `[-0.028, -0.008]` | `[1.0, 2.0]` | `PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest -q src/tac/substrates/sabor_boundary_only_renderer/tests` | SBO1 archive custody + scorer-free inflate + paired exact eval |
| 4 | `s2sbs_hf_byte_stuffing` | `[-0.025, -0.005]` | `[1.0, 3.0]` | `PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest -q src/tac/substrates/s2sbs_byte_stuffing/tests` | S2S1 archive custody + byte-consumption proof + paired exact eval |
| 5 | `driving_prior_pretrained_renderer_2032` | `[-0.012, -0.005]` | `[5.0, 30.0]` | `PYTHONPATH=src:upstream:$PWD .venv/bin/python tools/probe_driving_prior_readiness.py --output reports/cooperative_receiver/driving_prior_readiness.json` | dataset/license manifest + scorer penultimate-hook readiness + deterministic renderer export + paired [contest-CUDA]/[contest-CPU] exact eval |
| 6 | `driving_prior_world_model_substrate` | `[-0.012, -0.003]` | `[1.0, 3.0]` | `PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest -q src/tac/substrates/driving_prior_world_model/tests` | DPW1 byte-consumption proof + trained prior export + paired [contest-CUDA]/[contest-CPU] exact eval |
| 7 | `h15_coord_mlp_residual_sidecar_pr103_on_pr106` | `[-0.02, -0.005]` | `[1.0, 3.0]` | `PYTHONPATH=src:upstream:$PWD .venv/bin/python tools/probe_coord_mlp_residual_sidecar.py --output reports/cooperative_receiver/coord_mlp_residual_sidecar_probe.json` | sidecar bytes changed and consumed + full-frame parity baseline + paired [contest-CUDA]/[contest-CPU] exact eval |
| 8 | `s7_donoho_hnerv_wavelet_threshold` | `[-0.02, -0.01]` | `[0.0, 0.1]` | `PYTHONPATH=src:upstream:$PWD .venv/bin/python tools/build_cooperative_receiver_campaign_queue.py --output reports/cooperative_receiver/campaign_queue.json` | state_dict_parity + full-frame inflate parity + paired exact eval |
| 9 | `l2_sar_coherent_pose_spectrum` | `[-0.006, -0.002]` | `[0.0, 1.0]` | `PYTHONPATH=src:upstream:$PWD .venv/bin/python tools/probe_cooperative_receiver_pose_spectrum.py --pose-targets experiments/posenet_targets.bin --output reports/cooperative_receiver/pose_spectrum_l2.json` | spectral concentration smoke + pose codec roundtrip + paired exact eval |
| 10 | `n4_video_keyed_codebook` | `[-0.005, -0.001]` | `[1.0, 5.0]` | `PYTHONPATH=src:upstream:$PWD .venv/bin/python tools/build_cooperative_receiver_campaign_queue.py --output reports/cooperative_receiver/campaign_queue.json` | deterministic codebook regeneration + archive-byte reduction + exact eval |
| 11 | `b1_atick_redlich_segmap_eigenmodes` | `[-0.015, -0.005]` | `[0.5, 1.0]` | `PYTHONPATH=src:upstream:$PWD .venv/bin/python tools/build_lapose_foveation_atom_manifest.py --help` | basis custody + decoded mask/pose coupling proof + paired exact eval |
| 12 | `g5_mallat_scattering_decoder` | `[-0.04, -0.02]` | `[3.0, 5.0]` | `PYTHONPATH=src:upstream:$PWD .venv/bin/python tools/build_cooperative_receiver_campaign_queue.py --output reports/cooperative_receiver/campaign_queue.json` | trainer + archive grammar + runtime dependency closure + paired exact eval |
| 13 | `a1_plus_lapose_composition` | `[-0.013, -0.003]` | `[0.3, 3.0]` | `PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest -q src/tac/substrates/a1_plus_lapose/tests` | sidecar byte-consumption proof + predicted packet below 0.19 target + paired [contest-CUDA]/[contest-CPU] exact eval |
| 14 | `a1_plus_wavelet_residual_retarget` | `[-0.006, -0.001]` | `[0.1, 1.0]` | `PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest -q src/tac/substrates/a1_plus_wavelet_residual/tests` | score-sensitive wavelet atom selection + packet predicted below 0.19 + paired [contest-CUDA]/[contest-CPU] exact eval |
