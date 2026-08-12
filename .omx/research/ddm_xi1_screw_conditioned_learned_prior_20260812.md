# ddm_xi1 — screw-conditioned learned prior receipt

**Status:** Leg B complete; Leg A apparatus and deterministic contexts complete, but joint training is blocked because this governed sandbox exposes no Metal device. Frontier **UNMOVED**. No scorer was loaded and no score was claimed.

## Conclusions

1. CP135 already uses a useful learned/fitted prior: its fixed-point CAP1 AR(1)+bias packet is **22,242 B**, which is **85 B smaller** than direct CPR1 on the same current object. Direct storage is therefore not vindicated as a family.
2. A self-contained counted linear prior conditioned on a `tac.lie` relative screw is **30,072 B**, **7,745 B larger** than direct CPR1 and **7,830 B larger** than CAP1. The dominant honest tax is the **7,200 B** decoded geometric-pose plane, because CP135 does not carry geometric ξ.
3. Every Leg B packet decoded through its real implementation to the same canonical CPR1 bytes, SHA-256 `709ea928c2d73c599a9cffa85d9ea4f4cedab2940594f4b8ca39e4c60fd3a1d4`. The inherited CP135 contest-CUDA printed-8dp `d_pose = 0.00000688` therefore remains matched across the three rows.
4. Leg A has no rate verdict. PyTorch reports MPS built but unavailable, and MLX independently reports `No Metal device available` in this sandbox. CPU substitution was refused because the attested CL1 trainer is MPS-only. FA is not adjudicated.

## Leg A — matched learned-context byte table

No coded-byte row was produced. Empty cells are explicit unmeasured outcomes, not zero-byte results.

| λ rung | spatial-only Range bytes | spatial+ξ Range bytes | ratio | outcome |
|---:|---:|---:|---:|---|
| 1.0 | UNMEASURED | UNMEASURED | UNMEASURED | `READY_BLOCKED_NO_METAL_DEVICE` |
| 0.5 | UNMEASURED | UNMEASURED | UNMEASURED | `READY_BLOCKED_NO_METAL_DEVICE` |

The ready screen is seeded stratified-random n120: 10 temporal strata × 12 random frames, seed `20260812`. Both treatments use the same PR130 `IntegerHPAC`, initialization, λ-specific schedule, seed, 20 epochs, EMA law, capacity, packer, and Range coder. “Spatial-only” means the existing `conv_past` capacity receives a retained zero plane; “spatial+ξ” means that same input receives the retained ξ-warped prior, so capacity remains matched.

Retained preparation:

- spatial zero context: 23,593,088 B, SHA-256 `869d0bbeb19a9a12ca0fbb79e64f140e29fa9203a10d8058af2cd2296d573a82`
- ξ-warped context: 23,593,088 B, SHA-256 `8dc5db7125115362652b02d36adeeeb868d3e6a0d592d84b2a0ed85061215dd4`
- deterministic ξ repeat: 23,593,088 B, the same SHA-256
- preparation receipt: `/Volumes/APDataStore/pact/ddm_xi1_20260812/LEG_A_PREPARATION.json`
- exact fire order: `/Volumes/APDataStore/pact/ddm_xi1_20260812/queue/leg_a_mps.json`

Axis: `[macOS-MPS research-signal training; macOS-CPU advisory real Range bytes; scorer-free]`. The training/Range portion is unmeasured because Metal admission failed before training.

## Leg B — pose bytes versus matched realized d_pose

| coding on the CP135 current object | exact packet bytes | matched realized d_pose | canonical CPR1 decode | delta vs direct |
|---|---:|---:|---|---:|
| direct CPR1 + unchanged selector | 22,327 | 0.00000688 | exact | 0 B |
| incumbent CAP1 AR(1)+bias + Rice + selector | 22,242 | 0.00000688 | exact | −85 B |
| counted linear ξ model + Rice + 7,200 B ξ plane + selector | 30,072 | 0.00000688 | exact | +7,745 B |

Axis: `[macOS-CPU advisory; n600 exact carrier decode; inherited contest-CUDA CP135 d_pose]`. The d_pose value is inherited only because all three lossless packets restore the exact current CP135 CPR1 carrier bytes; it was not rescored in this arm. Packet bytes are real serialized n600 bytes, not entropy estimates. Each packet, repeat, decoded CPR1, and result receipt is retained under `/Volumes/APDataStore/pact/ddm_xi1_20260812/retained/leg_b/` with bytes and SHA-256.

The learned packet’s counted model is 792 B: six float32 screw scales, a 30×12 int16 weight matrix, and twelve int32 biases. Its input is previous coefficients, previous coefficient delta, and `log(inv(T[t-1]) @ T[t])` from the decoded 600×6 fp16 pose plane through `tac.lie`. The residual is real Rice code and the decoder reconstructs CPR1 exactly.

## Falsifiers

- **FA: NOT ADJUDICATED.** Neither learned context arm reached training or Range coding. The scoped negative is `ENVIRONMENT`: no Metal device in this sandbox. It is not evidence against the ξ-conditioned learned-context family.
- **FB: DOES NOT FIRE.** The rule required both dynamics formulations to be at least direct bytes. CAP1 AR is 85 B smaller than direct on the current object, while the self-contained geometric-ξ learned formulation loses. Verdict scope: `FORMULATION` for the counted linear geometric-ξ model; direct storage is not family-authoritative.

## RECALL EVIDENCE

The recall census preceded design and used content searches, not only charter filenames.

- Searched `.omx/research/`, the canonical research indexes and `sub015_DAG_*` FEED blocks, `.omx/state/main_hot_state.md`, and task/queue stores with queries including `screw|xi|se3|warp|temporal`, `HPAC|conv_past|CL1|SR1`, `CAP1|CPR1|AR1|B-spline|pose prior`, and `PK2|PZ4A|AM1|TF1`.
- Ran `tools/list_canonical_equations.py --json` and inspected `pose_ego_screw_twist_identifiable_up_to_affine_v1`, `ego_motion_cumulative_se3_bspline_v1`, and `partition_temporal_transport_amortization_jitter_bound_v1`.
- Inspected the actual CL1 trainer and attested receipts under `/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/`. Beyond the charter seeds, the existing CL1 model was found to consume the **unwarped previous decoded partition** through `conv_past`; this changed the matched spatial control to a zero plane with identical architecture/capacity.
- Inspected HM1’s seeded 10×12 n120 real-Range apparatus. This changed Leg A from a contiguous prefix to the legal stratified-random selection and preserved an explicit 5× projection boundary in the runner.
- Inspected CP135 parse-back/runtime and PR135 ExperimentBook CAP1 sources. Beyond the charter seeds, CP135 was found to **already use CAP1 AR(1)+bias** and to carry a photometric carrier rather than geometric ξ. This changed Leg B to count the 7,200 B decoded ξ plane in the learned row instead of granting it as free side information.
- Recalled PK2’s older-object exact AR/order/spline rows and PZ4A/PZ4R’s gauge receipts. They changed the verdict boundary: PK2’s older PR130 overlay negatives were not transferred to the current CP135 object, and PZ4A’s output-gauge rank closure was not treated as a lossless coefficient-codec result.

## Apparatus and verification

- Runner: `tools/run_ddm_xi1_screw_conditioned_learned_prior.py`
- State: `/Volumes/APDataStore/pact/ddm_xi1_20260812/state.json`
- Leg B receipt: `/Volumes/APDataStore/pact/ddm_xi1_20260812/LEG_B_RESULT.json`
- Leg A preparation: `/Volumes/APDataStore/pact/ddm_xi1_20260812/LEG_A_PREPARATION.json`
- Queue dispositions: `/Volumes/APDataStore/pact/ddm_xi1_20260812/QUEUE_DISPOSITIONS.md`
- Static checks: Ruff clean, `py_compile` clean, payload-retention strict gate 0 findings, inline deterministic self-test PASS.
- Review tracker: two sealed post-edit passes covered all 46 Python entities; policy check reported 46 compliant and 0 violations. No override was used.

The runner is stage-resumable. Each Leg A cell preserves an initial checkpoint, every epoch, the continuous-stage end, the QAT-stage end, and a latest continuation pointer with live weights, EMA shadow, optimizer, scheduler, and RNG state. Each materialized model, code lattice, symbol plane, Range payload, repeat, and decoded symbols is retained.

## Follow-on dispositions — verbatim

- ddm_xi1_leg_b_runtime: FOLDED. Owner: ddm_xi1. Consumer store: /Volumes/APDataStore/pact/ddm_xi1_20260812/LEG_B_RESULT.json. Fire trigger: none; CAP1 is already the CP135 incumbent and counted geometric-xi conditioning did not beat direct storage.
- ddm_xi1_leg_a_mps: QUEUED-WITH-A-FIRE-ORDER. Owner: MAIN Metal executor. Consumer store: /Volumes/APDataStore/pact/ddm_xi1_20260812/LEG_A_RESULT.json. Fire trigger: a governed process reports torch.backends.mps.is_available() == True; execute the pinned resume_command without CPU substitution.

## Frontier

No exact candidate was built or evaluated, so the pointer did not move. Current effective frontier remains CP135 `S = 0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]`, archive SHA-256 `6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6`. Current own-vehicle frontier remains LC2 `S = 0.16959899569230852 @ 187,226 B [macOS-CPU advisory, n600]`.
