# PACT compiler dashboard (Vehicle-OS)

Generated 2026-06-10T00:27:24.200675+00:00 from `/Users/adpena/Projects/pact`. Per `docs/vehicle_operating_system.md` Dashboard discipline: work is dashboard-driven; no stale-memory decisions.

## Frontier (pointer-only — never hardcoded)

- contest-CPU: **0.19198533626623068** (`b7106c9bdbb8`)
- contest-CUDA: **0.20533002902019143** (`9cb989cef519`)
- submitted PR for current frontier: `None`
- pointer last refreshed: 2026-06-09T07:07:25.354047+00:00

## Per-vehicle maturity (L0-L7)

| vehicle | L | allowed_claim | latest_artifact (sha) | authority_tier | metric_family | current_blocker | next_command | owner | pass_route | fail_route |
|---|---|---|---|---|---|---|---|---|---|---|
| snerv | L4 | exact_scored_row_exists | /Volumes/VertigoDataTier/pact/snerv_mistake_b_g1a_20260609T201221Z/snerv_g1b_export_binding_verdict.v1.json (AUDIT_PENDING) | exact_cpu_advisory | exact_pair_scorer | 100% rate (581.6MB skip_high float64 LL=99.9996%); export bound but rate chasm -> route to LF entropy-coding front; NOT L5 (no contest-axis paired row, pays_rent=false) | .venv/bin/python tools/... (LF entropy-coding rate attack; branch B ladder on snerv_branch_b_rate_attack) | snerv_branch_b_rate_attack | LF entropy-coding front lowers rate -> V3 judges first real program -> V5 atoms stack | store-LF rate chasm is the research front (entropy-code the LF; SNeRV store-LF/generate-HF) |
| hi_nerv | L1 | mechanism_present_unit_tested | /Volumes/VertigoDataTier/pact/hinerv_codec_counterfactual_ablation_smoke_20260608T_v9/hi_nerv_receiver_closed_modelsize_ladder.json (AUDIT_PENDING) | AUDIT_PENDING | AUDIT_PENDING | objective severance at weight surface | .venv/bin/python -m pytest src/tac/substrates/hi_nerv/ (then vendor PR95-HNeRV port vs patch decision per #40) | hinerv_completion (task #40) | vendor-faithful PR95-HNeRV port OR patch (decided by #45 + atlas omega) | skip recon-inert under MSE; needs grid-PE/skip ON + reachable objective |
| pact_nerv_vq | L1 | mechanism_present_unit_tested | /Users/adpena/Projects/pact/.omx/state/vehicle_fidelity/pact_nerv_vq.json (AUDIT_PENDING) | AUDIT_PENDING | AUDIT_PENDING | objective severance at weight surface | audit MLX-route scorer-objective VJP; set nonzero SegNet/PoseNet weights; skip+omega under scorer objective (NOT MSE) | pact_nerv_vq_completion (task #44) | skip + omega(measured) under scorer objective (NOT MSE); audit MLX-route VJP | skip-free decoder mean-fields regardless of VQ; architecture under scorer-objective untested |
| sane_hnerv | L0 | research_carrier_sketch | /Users/adpena/Projects/pact/.omx/state/vehicle_fidelity/sane_hnerv.json (AUDIT_PENDING) | AUDIT_PENDING | AUDIT_PENDING | docstring claims a mechanism the forward never implements | implement advertised bilinear-skip in forward OR correct the docstring (architecture.py:5,27,122) | laundering remediation | implement the advertised bilinear-skip OR correct the docstring | documentation-fake: docstring claims bilinear-skip the forward never implements |
| ff_nerv | L0 | research_carrier_sketch | /Users/adpena/Projects/pact/.omx/state/vehicle_fidelity/ff_nerv.json (AUDIT_PENDING) | AUDIT_PENDING | AUDIT_PENDING | no HF residual path / objective mechanism present | (dormant) add a genuine HF residual path before any training run | (dormant sketch) | add a genuine HF residual path; 64x64 DCT grid cannot represent boundary HF | skip-free + band-limited by construction -> strictly-worse mean-field variant |
| pr110pp | L3 | archive_real_byte_closed_consumed | /Users/adpena/Projects/pact/experiments/results/pr110pp_r1plus_candidate_20260609/candidate_archive.zip (AUDIT_PENDING) | exact_cpu_advisory | advisory_pose_delta | L4 exact row in flight (R1 paired contest-CPU Modal eval dispatched); advisory pose gain only | .venv/bin/python tools/... harvest R1 paired contest-CPU Modal eval (strict_candidate_cpu/modal_call_id.txt) -> exact CandidateActionEvaluation | pr110pp_r1_paired_eval / pr110pp_r2_nonmps_mode_table | R1 paired contest-CPU eval confirms dS<=0 -> exact CandidateActionEvaluation row (L4) | macOS-CPU vs Linux-x86_64-CPU per-mode ordering unstable -> no exact gain |
| atlas_atoms_v3 | n/a-vehicle | infrastructure (not a candidate program generator) | AUDIT_PENDING (AUDIT_PENDING) | n/a | n/a | n/a — infrastructure | atlas v2 full sweep (running) -> learned-omega Nyquist-capped basis; V3 ingest of #45's row | atlas_engine_mlx_jacobian / frozen_evaluator_contract | measured law geometry + DeltaS-judge feed every vehicle's search/accept | n/a — infrastructure, not a candidate program generator |

## Maturity evidence (FROM EVIDENCE — cite per assignment)

- **snerv** L4: exact CandidateActionEvaluation: candidate_action_evaluation_g1b_pathb_ep273.v1.json: d_seg=0.0024678972016166276 d_pose=0.002033528243975032 bytes=581583207 pays_rent=False; fidelity clean (MFU/HFR/TUB+DWT), reachability clean (VJPs reach)
- **hi_nerv** L1: vehicle_fidelity present mechanisms ['bilinear_skip'] (file:line evidence); L1 mechanism-present
- **pact_nerv_vq** L1: vehicle_fidelity present mechanisms ['codebook_vq'] (file:line evidence); L1 mechanism-present
- **sane_hnerv** L0: NAME-LAUNDERING (documentation-fake): NAME-LAUNDERING: docstring claims 'bilinear_skip' (phrase(s): ['canonical HNeRV with per-pair latent + bilinear-skip + sin activation', '(with bilinear-skip from each prior block)', 'One Conv -> sin -> PixelShuffle(2) block, with optional bilinear-skip']) but it is in mechanisms_absent — the carrier advertises a mechanism it does NOT implement (Catalog #307 documentation-fake).
- **ff_nerv** L0: vehicle_fidelity manifest: zero present mechanisms (honest sketch)
- **pr110pp** L3: byte_closure_proof.json + noop_detector.json(consumption_proven=true) at /Users/adpena/Projects/pact/experiments/results/pr110pp_r1plus_candidate_20260609/candidate_archive.zip
- **atlas_atoms_v3** n/a-vehicle: spectral atlas (evaluator_response_atlas.py) + V3 (frozen_evaluator_contract.py) are the measured-law + DeltaS-judge kernels every vehicle consumes; not on the L0-L7 ladder

## Live work (running daemons/agents)

| subagent_id | status | step | next_action | written_at_utc |
|---|---|---|---|---|
| snerv_branch_b_rate_attack_20260609 | in_progress | 7 | Full ladder N=48 in progress pid 32009: R0 reproduces G1b EXACTLY (d_seg=0.002468 pose=0.1426 total 387.642). Awaiting R1-R7 + 4 lf-floor rows. Then report+ingest+commit. | 2026-06-10T00:21:32.728812+00:00 |
| pr110pp_r1_paired_eval_20260609 | in_progress | 3 | Re-dispatch 3 Modal CPU evals with fixed wrapper (distinct per-job lane_ids); validate frontier baseline scores; then strict + r1plus; harvest; ingest exact rows | 2026-06-10T00:19:55.465189+00:00 |
| pact_dashboard_maturity_20260609 | in_progress | 2 | evidence gathered; write generator module + tests | 2026-06-10T00:19:37.513181+00:00 |
| atlas_engine_mlx_jacobian_20260609 | in_progress | 4 | Await 600-pair sweep completion (pid 62686, pair 75/600), fill memo headline, commit via serializer | 2026-06-10T00:18:58.657092+00:00 |
| snerv_g1b_export_binding_20260609 | in_progress | 10 | AWAIT advisory daemon (pid 94482) monitor event, then: .venv/bin/python /Volumes/VertigoDataTier/pact/snerv_mistake_b_g1a_20260609T201221Z/g1b_consolidate_verdict.py (one command). All decisive work committed (acbe76a7a). If session dies: successor runs the consolidation script + reads memo for full context. | 2026-06-09T22:40:58.610738+00:00 |
| b1_campaign_v2 | in_progress | 8 | write campaign ledger + register lane + review-gate + commit; record prereq-evidence blocker chain | 2026-06-09T04:51:56.793210+00:00 |
| b1_campaign | in_progress | 5 | PRIORITY 3 wired+validated (curriculum engages, blocker drops). Add wiring tests + run existing runner tests for regression + ruff; commit; then LAUNCH full curriculum. | 2026-06-09T04:10:12.292136+00:00 |
| hinerv_60kb_mps_fit_probe_20260602 | in_progress | 1 | Write probe script: build chosen HiNeRV cfg, MPS faithful fit (~300 steps), CPU-mirror measure d_seg/d_pose + packed rate | 2026-06-02T11:19:22.975426+00:00 |
| snerv_learned_hf_decoder_20260601 | in_progress | 5 | Write learned_trainer.py: MLX score-aware trainer (saliency-weighted DWT-domain HF objective via faithful MPS scorers + pose/seg saliency push; orthonormal Parseval => DWT-coeff loss == pixel loss) | 2026-06-01T22:01:05.479631+00:00 |
| inverse_steganalysis_phase1_carrier_wirein_20260601 | in_progress | 1 | read L-inf gate + score_exact_saliency + hprc_synthesis_adjoint + pr101 packer/codec + locate decoder state_dict; design G3 adjoint into HNeRV latent domain | 2026-06-01T20:19:23.177141+00:00 |
| z8_p18_p19_freeze_vs_implicit_kkt_comparison_20260531 | in_progress | 4 | Write >=10 NO-FAKE tests | 2026-05-31T17:59:16.652365+00:00 |
| z8_joint_p18_p19_deadzone_rate_attack_20260531 | in_progress | 12 | Committed ad73c2863 (impl+tests+lane); memo+MEMORY.md landed; 600-pair (pid 18962) building baseline; poll for completion to refine memo with full-scale numbers | 2026-05-31T17:48:37.445700+00:00 |

## Schema gaps

- **vehicle_fidelity_manifest.v1**: no maturity_level field — maturity cannot be recorded in-manifest via the canonical emitter; the dashboard derives it from fidelity + reachability + constants + typed verdict rows instead. _Remediation_: council/operator-approved schema addition (a maturity_level field on VehicleFidelityManifest) — a design decision, NOT a hand-edit; OR standardize on constants_provenance.declared_maturity_level for ALL vehicles (only hi_nerv has a constants manifest today).

## Notes

- Maturity is assigned FROM EVIDENCE (see each row's maturity_evidence); no level is asserted without a machine-readable basis.
- Scores are POINTER-ONLY (canonical_frontier_pointer.json); never hardcoded.
- SCHEMA GAP: vehicle_fidelity_manifest.v1 has no maturity_level field; the canonical per-vehicle declared maturity lives in constants_provenance (declared_maturity_level) where a constants manifest exists (only hi_nerv today). The dashboard derives maturity from the union of fidelity + reachability + constants + the typed verdict rows.
- snerv fidelity manifest id is 'snerv_inverse_steg_carrier'; reachability id is 'snerv' — bound via VehicleSpec.fidelity_id/reachability_id.

## Supplementary: carrier triage (closest-to-exact-archive first)

| vehicle | composition | readiness | score+axis | blocker | next route |
|---|---|---|---|---|---|
| hinerv | leaf_latents | has_exact_archive_now | 89.57 [macOS-CPU advisory] | carrier broken at fp16 too (fp16~int8, d_seg 0.50 / d_pose 151); live-MLX gate pending to  | already emits archive -> B2 bridge (done: rejected) |
| source_recode | source_recode | has_exact_archive_now | 0.19198533626623068 [contest-CPU] | orphaned from the neural carriers; not yet composed with sparse evaluator atoms in V3 | already the CPU frontier anchor (fp11_source_brotli_recode); compose with sparse atoms as  |
| pr110pp | selector_menu | scaffold_needs_training_or_export | — | differentiable selector x menu optimizer not built; comp-Muon-INSPIRED (not drop-in) partn | reproduce K=16 selector on frozen HNeRV -> Huffman selector stream -> B2 (the discrete for |
| atom | sparse_atoms | scaffold_needs_training_or_export | — | atoms only pay rent against a GOOD base; gated on a viable carrier (or on source_recode as | mine from inverse-steg cost map (seg margin field) + cooperative-receiver nullspace -> mat |
| snerv | source_state | scaffold_needs_training_or_export | — | source-forward causal proof + LF/HF byte-pressure binding incomplete; NO exact-eval row ye | TUB DROP_OR_REIFY source-forward proof -> MFU/HFR binding -> export -> B2 (must produce so |
| pact_nerv_vq | codebook | pending_audit | — | maturity unknown — audit running; if mature with archive path, invest HERE before HiNeRV r | PENDING pact_nerv_vq_maturity_audit (sibling lane that may already embody the codebook) |
| hinerv_codebook | codebook | design_only_no_archive_path | — | GATED on live-MLX: do NOT retrofit composition onto a carrier that is broken at fp16 | retrofit codebook into HiNeRV latents + export -> B2 (only worthwhile if live-MLX gate say |

## Supplementary: active training heartbeats

- heartbeat_b1_b1_229k_pilot_20260609T055851Z.log: 2026-06-09T08:21:36Z pid=92660 run=b1_229k_pilot_20260609T055851Z
