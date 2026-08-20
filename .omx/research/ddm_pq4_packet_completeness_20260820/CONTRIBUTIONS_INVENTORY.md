# Contributions inventory — every original mechanism, adaptation, optimization and port

`date_utc: 2026-08-20` · `owner: ddm_pq4` ·
`axis: the score row is [contest-CUDA T4, n600]; this document measures nothing and claims no score` ·
`score_claim: false` · `promotable: false`

**Own-vehicle frontier: S = 0.14839100138338618 @ 180,625 B `[contest-CUDA T4, n600]`,
archive `f3bce5d259a08183…`.** Unmoved by this arm.

---

## Why this document exists

`BORROWED_SUBSTRATE_ACCOUNTING.md` answers **"what is in the archive, and whose is it?"**
It is section-scoped by design, so a mechanism that shaped the candidate but does not own a
section — a solver, a law, an instrument, a port — has no row there and disappears.

This document answers the other half: **"what did we build?"** It spans four classes across
the whole programme, in-packet and research-only alike, and every row carries a receipt path.
Where the two documents overlap they must agree; §5 records the diff.

**The honesty rule is the accounting's.** A row with no receipt does not get `ORIGINAL`.
`IN-PACKET = YES` means the mechanism's effect is present in the `f3bce5d2…` bytes; it does
**not** mean the mechanism is solely ours — read the CLASS column for that.

**Five of this arm's own charter premises were falsified by recall.** They are listed in §6
rather than quietly corrected, because four of them would have produced an over-claim.

---

## Class 1 — ORIGINAL (designed and built in this repository)

| # | Name | Mechanism | Receipt | In packet | Measured |
|---|---|---|---|---|---|
| O1 | **Per-pair KEEP/DROP joint admission waterfill** | Each of 573 edited pairs has two measured states (DROP = base frame 1 + br1 codes, 0 tokens; KEEP = edited frame 1 + re-solved carrier). Admission swept over a Lagrange multiplier on pose damage, every subset scored through the exact contest formula — a fixed-ratio greedy is wrong at the margin because the pose leg is √-concave | `.omx/research/ddm_jg5_pose_resolve_on_edited_renders_20260819.md` §5; accounting §9.3 | **YES** | **455 of 573 admitted**; 0 counted archive bytes; net −0.00776976 S vs the br1 pointer |
| O2 | **Derived materiality-floor stop rule** | Replaces br1's `iterations=6` counter: iterate while `remaining_dd · dS/dd_i > DELTA_FLOOR_S`, with `remaining_dd` extrapolated from each pair's OWN geometric decay ratio. No hand tolerance survives | same as O1, §4 | **YES** (governed the shipped solve) | threshold **5.588639e-09** d_pose units. **Honest caveat: the rule never bound** — 600/600 stopped on `no_improving_step`, 0 budget hits |
| O3 | **Edits-are-a-pose-actuator law** | A seg-descending token-edit direction is not pose-null. Edits must be priced as a POSE actuator | same as O1, §1; `.omx/research/ddm_jg1_joint_solve_20260819.md` | **YES** as design rationale; 0 bytes | edits bought −0.012847 S seg, cost +0.172 S pose = **13.4× loss**; 571/573 pairs worse on pose |
| O4 | **jg2 tail re-encoder — the exact inverse of the shipping decoder** | `decode_production_tokens` line-for-line with decode replaced by encode, importing model / 190-group plan / boundary map / fixed table / corrector from the shipped runtime. Returns the **exact archive delta** for any edited token field instead of a bits-per-token constant | `.omx/research/ddm_jg2_sub015_chain_20260819.md` §S1b–S1i | **YES** — jg5 §6.2 re-encoded the 455 admitted edits with it | control byte-identical at 109,696 B; **3.8373 measured bits/changed token** at jg5 scale |
| O5 | **Edit-cost superposition law** | Token-edit RATE costs ADD; interactions under 3% and sign-changing | same as O4, §S1h | **YES** as pricing law (lets the waterfill sum per-chunk rate) | union/sum **1.0258**; 10+6+14 = **measured 30 exactly**; causality control 0.000000 bits |
| O6 | **Union ≠ sum-of-legs law (compensation axis)** | On the COMPENSATION axis the opposite regime holds — joint compensation beat the naive union | `.omx/research/ddm_bu1_bank_union_compile_20260817.md` | n/a (pricing law) | **3.705×**. Explicitly the opposite regime from O5; the two are not contradictory, they are different axes |
| O7 | **Cascade reach ≠ cascade magnitude** | A global cascade can be a CREDIT; dependency arguments bound WHERE an effect reaches, never HOW BIG | `.omx/research/ddm_jg2_sub015_chain_20260819.md` | n/a (pricing law) | 3 additivity laws, none transferred |
| O8 | **Container transform "plane2" (ck2, eleventh move)** | Parameter-free transform changing only how four already-decided section bodies are laid out before Brotli; the receiver restores each body byte-for-byte before parsing, so both distortion legs are zero **by construction** | `.omx/research/ddm_ck2_container_plane2_eleventh_move_20260819.md` | **YES** — ancestry 177,182 → 176,525 | **−657 B**, rate ΔS −4.374693e-04, d_seg 0, d_pose 0 |
| O9 | **Tail-override build step (to1, twelfth move)** | Adds the missing step that substitutes a **re-encoded** tail into the pointer body. `ddm_sa3_rebase_sz1.build_candidate` had always borrowed sz1's tail verbatim, so every token-stream rate win measured elsewhere was structurally unreachable from the pointer | `.omx/research/ddm_to1_tail_override_twelfth_move_20260819.md` | **YES** — to1 body 176,420 B `50e56145…` is the base up2/up3/br1/jg5 build on | **−105 B**, ΔS −6.991519e-05, pure rate |
| O10 | **Fixed-point integer log-odds mixer (fx1)** | Weighted geometric mean in log-odds space using **radicals rather than lookup tables** — transcendentals avoided entirely, because IEEE requires correctly-rounded `sqrt` but not `log`/`exp`. Escapes the `min ≤ mean ≤ max` hull that bounds every arithmetic-mean-in-odds rule me1 raced. Decode-identical | `.omx/research/ddm_fx1_fixed_point_logistic_mixer_20260817.md` §2, §9 | **YES** (see A7 for the attribution half) | **−560.07 B** on the n600 token field; byte-closed 180,601 B, ΔS −3.72881e-4; +127.3 s decode |
| O11 | **Within-miss relative law / miss-sector rung (ma1)** | Per-cell multiplicative correction on the miss class, online, decode-identical. Also **withdrew** its own "77,241 B reservoir" framing as a vacuous denominator | `.omx/research/ddm_ma1_model_axis_miss_cost_20260819.md` | **YES** — the shipping tree's `runtime/free_corrector.py` **is** `Ma1WithinMissCorrector` (confirmed at source by wc2c §3) | **−104.584 B**; projected archive −105 B, ΔS −6.9915e-05, 20.0× the admit bar. Reservoir corrected 77,241 B → **~400 B + ~180 B** |
| O12 | **Uncapped exact pose GN solve (up2)** | Ran the carrier coordinate descent uncapped to a convergence proof on all 600 pairs at zero archive bytes | `.omx/research/ddm_up2_shipping_object_pose_solve_20260819.md` | **YES** | d_pose 7.769484e-06 → **7.649247e-06**; 429 improved / **0 worsened**; **ΔB = 0** |
| O13 | **Un-interleave discovery + Rice-payload splice (up3, thirteenth move)** | Both of up2's byte-close blockers were ONE missing transform: the stored carrier is 2-plane byte-interleaved (`reserved` bit `0x04`) and the receiver un-interleaves at `residual_archive.py:188` before reading any offset. After un-interleaving, byte 139 reads the true `k_base` and the Rice payload sits at `6+96+40+basis_bytes` | `.omx/research/ddm_up3_thirteenth_move_byteclose_20260819.md`; `experiments/ddm_up3_carrier_splice.py::build_archive` | **YES** — the splice tool jg5 §6.3 uses for every level it builds | byte-closed 176,420 B at **ΔB = 0**; realized d_pose matched prediction to 8 sig figs. Also corrected a stale pin: `PACKED_CAP1_SECTION_BYTES` 22,183 → **22,178** |
| O14 | **Damped Gauss-Newton carrier solve (br1)** | The residual demands a **multi-coordinate** step of 57 to 14,079 int12 code units; up2's single-coordinate ±2 search could never travel that far. Replaces the search with a damped GN step on the same basis, lattice and bytes | `.omx/research/ddm_br1_pose_basis_reorientation_20260819.md` | **YES** — br1 `44e9e650…` @ 176,429 B is jg5's pointer body; its codes are the DROP-branch carrier for the dropped pairs, and `gn_solve_pair` is jg5's solver reused verbatim | d_pose → **6.993157e-06** at ΔB **+9**; net ΔS −3.774950e-04 = 107.9× the admit bar; 204/600 improved, **0 worsened** |
| O15 | **jg4 re-encoder checkpoint fix** | `encode_tail` saved the coder's model via `corrector.state_dict()`, defined only on the flat rr4 base class. The shipped corrector is three subclasses deeper and neither overrides it, so a resumed run restarted the model-mixing half **cold** while every log line looked healthy | `.omx/research/ddm_jg4_reencoder_mirror_fix_20260819.md` | **YES as a correctness precondition** — jg5's re-encode is `delta_trustworthy=true` because of it | checkpoint saved **7 of 97** values, dropping **9.68 MB** of live state. Resume divergence: straight-through byte-identical; resumed-at-275 **+127 B** |
| O16 | **Candidate-seal contract** | One typed document pinning archive / member / runtime FILES digest / per-file receiver pins / `admit_bar` **with its derivation** / axis / retained payloads / pre-registered falsifiers / `seal_sha256`; a validator re-derives every pin from disk at consumption; the firer refuses a paid call on drift. Every sealed value is REMOVED from the fire command line | `.omx/research/ddm_seal1_candidate_seal_contract_20260818.md`; `src/tac/candidate_seal.py` | **YES** — packet seal `96e9860a…`, `SEAL_VALID` at fire time | 36 new + 49 sister controls green; 9 typed verdicts. Each field extincts a named measured failure (rr2's S 27.83; qs4's +2.4e-4 S) |
| O17 | **Canonical contest-score arithmetic** | One helper for `100·d_seg + √(10·d_pose) + 25·bytes/37,545,489`, byte-identical to `upstream/evaluate.py:92`, replacing ~20 hand-rolled copies; plus `break_even_d_seg` and an upstream parity test | `src/tac/contest_score.py`; `.omx/research/canonical_contest_score_helper_compliance_hardening_20260623.md` | **YES as arithmetic** — pq3 re-derived the packet score from all three components; also the substrate for O2's `dS/dd_i` | the bug it cures: a dropped ×25 claimed break-even d_seg 1.89e-3 where correct is **~7.8e-4** |
| O18 | **Canonical packet stager** | Manifest-driven copy, re-hash after copy, tree hash re-derived from the staged rows, census with its denominator, fail-closed with output removed on mismatch. Replaced four ad-hoc per-generation stagers | `tools/stage_contest_submission_packet.py`; `.omx/research/ddm_pq3_packet_rebase_jg5_20260820.md` | supports the packet | 33/33 rows re-hashed. **Has no tests — owed** |
| O19 | **Packet census guard** | Reports its denominator and cannot exclude a file class from both sides of its own comparison — the exact defect generation 4 recorded in itself | `tools/packet_census_guard.py`; `gen5_receipts/CENSUS_gen5.json` | supports the packet | `CENSUS_CLEAN` / `PREP_CLEAN` / `RECEIPTS_CLEAN`; caught 51 AppleDouble sidecars with exact paths |
| O20 | **Canonical dispatch firers** | Modal exact-eval and local advisory fires go through one reviewed entry point each | `tools/fire_modal_auth_eval.py`; `tools/fire_local_advisory.py` | supports the row | — |
| O21 | **RC64 backend role registry** | Hashed all 241 `rc64_backend.c` copies across three custody roots and separated them by ROLE | `reverse_engineering/rc64_backend_role_registry.json` | n/a (custody apparatus) | 241 files, **4 distinct contents**, one of them a third party's PR #138 body |
| O22 | **End-to-end compression entry point** | One sanitized command that rebuilds the archive from pinned retained inputs and refuses to exit 0 unless the bytes hash to the pin; no filesystem layout in the file | `experiments/ddm_pq2_compress_e2e.py` | mechanism ships as documentation; **NOT re-run for these bytes, and structurally CANNOT rebuild them** — §4 | rebuild verified 2026-08-17 for the rr4 candidate only |
| O23 | **Receiver binding, archive assembly, custody chain** | Binding our sections into a contest-legal archive plus its provenance chain | accounting §2 row 9 | **YES** | runtime tree `2103073d…` |
| O24 | **Carrier framing runtime patch** | `DDM_SA2_VARIABLE_PACKED_CAP1_V1` derives the packed-CAP1 section length from the section's own u24 bit counts instead of a pinned constant; generic framing, rule-118 clean | accounting §8.4; `runtime/residual_archive.py` | **YES** | `counted_bytes: 0` |
| O25 | **Dual-axis eval workers** | A governed T4 materializer that runs each runtime's unmodified `inflate.sh`, decodes GT on the exact upstream DALI batch-16 surface, and retains every GT RGB batch, SegNet input, logit tensor and argmax field — **both scoring axes off one dispatch**. The local consumer is scorer-free and refuses to decompose unless remote controls read exactly the pinned flip counts | `.omx/research/ddm_js1b_cuda_argmax_field_materializer_20260813.md`; `.omx/research/ddm_js1b_cuda_custody_adjudication_20260813.md` | **NO as bytes / YES as ancestry evidence** — the mc36 union it adjudicated is ancestry step 4 | 712.0 s T4, ~$0.16, both axes. Quantified the local-renderer drift it exists to avoid: Mac-CPU decode inflates flips **+44%** (cp135) and **+75%** (t1r1_c1), **non-uniformly across archives** |
| O26 | **Level-set / task-space witness research line** | Code the task-sufficient statistic (argmax partition + ego-pose) rather than RGB; stratified screw-warped level-set factorization with an MDL-canonicalization gauge that canonicalizes for RATE over a task-equivalence class | `docs/paper/novel_contributions_and_originality_accounting.md` (Ledger 2); `.omx/research/levelset_witness_trainer_landed_20260627.md`; `.omx/research/CAPSTONE_witness_taskspace_roundtrip_byte_floor_formulation_20260621.md` | **NO — research-only**; this candidate is not a witness vehicle | Fisher curvature ↔ class margin Pearson **0.978**; ~97% of d_seg in ~4.7% of frame area; nonlinear intrinsic dim ≈8–9; indirect-RD task floor S_floor ≈ 0.118 |

## Class 2 — ADAPTATION OF LINEAGE (their idea or source, our implementation / re-fit)

| # | Name | Inherited vs ours | Receipt | In packet |
|---|---|---|---|---|
| A1 | **PR #130 `semantic-pose-HPAC_CPR1`** (Fesal Fayed, `fesalfayed`) | The origin vehicle at 191,052 B, taken under the off-the-shelf grant. Nothing about it is claimed | `.omx/research/pr86_pr130_fullstack_intake_20260728.md`; `.omx/research/pr130_eureka_intake_acquisition_20260806.md` | **YES** (ancestry root) |
| A2 | **PR #135 `semantic-pose-HPAC_CPR1_polished`** (Shreyan Mohanty, `codexblack`) | The trained state this submission re-represents (186,724 B) plus the edit-then-recompensate pattern | `.omx/research/ddm_pi135_pr135_intake_20260810.md` | **YES** |
| A3 | **PR #133 `cpr1_cbq_matched8`** (`JasonMo123`) | Transitively in the ancestry through PR #135. Never taken directly | `.omx/research/ddm_pi135_pr135_intake_20260810.md:11` | **YES** (transitively) |
| A4 | **Semantic renderer state** | PR #135's trained values in OUR format, **lossily changed**. No byte-identity claim | accounting §8.1 | **YES** |
| A5 | **Pose carrier state** | Their solver form, our binding, their lattice re-solved | accounting §8.1 | **YES** |
| A6 | **HPAC probability object** | PR #130's architecture, retrained here on our own label field, checkpoint ep0634 chosen from 81 retained candidates by distortion-aware selection. §9.2 classes the SHIPPED object `inherited-substrate` | `.omx/research/ddm_hv1_harvest_compose_ep508_20260815.md`; `.omx/research/ddm_rx2_mc36_label_hpac_20260814.md` | **YES** |
| A7 | **Log-odds context mixing (the idea under O10/A8)** | The formulation and the context design are ours (O10); the log-odds context-mixing IDEA is the **PAQ lineage (Matt Mahoney)**, published in this contest first by **PR #138** and in weaker form by **PR #136** | accounting §7.1, §4 | **YES** |
| A8 | **Free decode-time probability corrector (rr2)** | Our per-group statistical estimator over the HPAC model's probabilities, zero archive bytes. Encoder proven the exact inverse of the shipping decoder. **Mechanism class published first by PR #138 `opal_v1`** — no priority claimed | `.omx/research/ddm_rr2_encoder_byteclose_20260817.md`; `.omx/research/ddm_rr1_free_decode_model_and_rate_rung_close_20260817.md`; concurrency table accounting §3 | **YES** — landed on the pre-registered target with **0 B deviation**; −1,598 B, ΔS −1.0640426e-3 |
| A9 | **Causal-geometry template widening (fx2)** | The shipped 190-group wavefront is exactly `group(x,y) = (x&63) + 2·(y&63)`; under it UP-RIGHT is as causal as UP and more so than LEFT. Widening to four neighbours plus a local-homogeneity feature is free receiver code | `.omx/research/ddm_fx2_model_axis_all_sections_20260818.md` | **YES** (13-member build) — **−710.84 B** shipped row → 180,450 B. UP-RIGHT causality 98.6945%. **SSE/APM measured dead: loses in 6 of 6 formulations** |
| A10 | **In-compile frame-0 Schur pose compensation** | Exact frame-0 signed-int12 moves cancel a frame-1 edit's PoseNet-6 leakage, as a Schur-coupled solve folded into the existing Rice lattice. **The edit-then-recompensate pattern is PR #135's, not ours.** Ours: the content-fingerprint binding that fails closed, the frame-0/frame-1 disjointness argument, the step-matched Jacobian, the rate route | `.omx/research/ddm_qs1_frame0_schur_coupled_solve_20260813.md`; `.omx/research/ddm_qs5_resolve_compensation_20260813.md`; accounting §8.2 | **YES** (§2 row 5) — pose-energy cancellation to **99.995054%**; rate route turns ~7,000 B of sidecar into **41 B** |
| A11 | **sa3 / keep01 compensated edits (8th + 9th moves)** | Mass-axis compensated-edit rebase onto the sz1 body | `.omx/research/ddm_sa3_compensated_edit_rebased_verdict_20260818.md`; `.omx/research/ddm_keep01_ninth_pointer_move_verdict_20260818.md` | **YES** (ancestry 179,930 → 179,140 → 177,576) — both carry ⚠ 2026-08-19 margin corrections (16.1×→**7.94×**; 164.9×→**80.4×**) from the summed-two-row-8dp-bound law |
| A12 | **SM3R mode-6 row-pruned mixed-depth semantic quantization** | Structured pruning and mixed-precision quantization are standard practice, on PR #135's tensors. Ours: the SM3R/SD1M wire formats, the surviving-marginal measurement, the fail-closed receiver integration | accounting §8.2 | **YES** — the generation where rows 1 and 2 **lost byte-identity to PR #135**; 6,713 of 7,200 carrier coordinates changed |
| A13 | **rr4 CUDA-prob re-encode** | **The chartered mechanism was FALSIFIED.** CPU-vs-CUDA divergence in the AR-prior probabilities was disproved from two existing receipts (identical logit and CDF shas across T4-CUDA-x86_64 and macOS-arm64-CPU). A different real defect was fixed: the `q = 1/(1+exp2(−(log2(p/(1−p))+δ)))` round trip, since IEEE does not require correctly-rounded `log2`/`exp2` | `.omx/research/ddm_rr4_cuda_prob_reencode_20260817.md`; `.omx/research/ddm_rr2_t4_refusal_device_scoped_decode_identity_20260817.md` | **YES** (ancestry 181,161 B). Round trip perturbs **50.086%** of positions by ~1 ULP; the differential test **refused to convict** (0 changed coding rows). **True cause of the T4 refusal remains OPEN** |
| A14 | **Micro-edit engine (me1)** | Engine to enumerate and price micro-edits. Its own finding: the semantic micro-edit families are at their measured asymptote, so it redirected to the coder axis | `.omx/research/ddm_me1_micro_edit_engine_20260817.md` | **NO directly** — its four raced context architectures all **lost** (best +359 B). Its CRUX — an average can never beat its best member — is what fx1 built, and fx1 IS in the packet |
| A15 | **Joint `{edit, drop, keep}` three-way proposal** | jg2's S2 spec clause 1. **`drop` is NOT SHIPPED**: jg3 declared it a MECHANISM reduction with cause — rc4's drop is a RECEIVER change and the pointer body's `cpr1/inflate.py` has no such path, so implementing it invalidates the byte-identity control chain the seal rests on | `.omx/research/ddm_jg2_sub015_chain_20260819.md`; `.omx/research/ddm_jg3_s2_joint_solve_20260819.md` OPTIMAL FORM table row 1 | **PARTIAL — `edit` + `keep` only.** `drop` is **owed headroom** priced at −0.002929 S (44.9% of the gap) |
| A16 | **Compressed model container, residual payload, table codes, RC64 encoder backend** | Inherited unchanged; no originality claimed. Residual-payload provenance remains **UNRESOLVED**, claim withdrawn | accounting §2 rows 3, 6, 8a; §6 item 1 | **YES** |
| A17 | **RC64 shipped receiver member** | PR #135-derived and **modified**; stated explicitly so the difference is mistaken for neither originality nor a clean copy | accounting §2 row 8b; sha `05839d14…` | **YES** |

## Class 3 — OPTIMIZATION OF LINEAGE, and one MEASURED NULL

| # | Name | Mechanism | Receipt | In packet | Measured |
|---|---|---|---|---|---|
| P1 | **cp135 lossless recompose** | Recomposed PR #135's own sections losslessly | `.omx/research/ddm_cp135_rate_compose_20260810.md` | ancestry | 186,724 → 186,252 B |
| P2 | **mc36 Variant C admitted micro-edit union** | qs2 ∪ re1 edit union, promoted on T4 | `.omx/research/ddm_mc36_promotion_complete_s_verdict_20260814.md` | **YES** — compensation pairs `[7, 96, 105, 176, 178, 517, 523]` carried in the base | 186,269 B; best-ever realized semantic micro-edit at −2.068e-5 |
| P3 | **12-dim basis re-orientation — MEASURED NULL** | Re-mixing the 12 stored basis dimensions leaves the reachable pose correction invariant | `.omx/research/ddm_br1_pose_basis_reorientation_20260819.md`; `.omx/research/ddm_br1_basis_race_and_drop_surface_20260803.md` | **NO — the re-orientation ships nothing.** What shipped from br1 is the GN solve (O14) | invariant to **1.9e-08** (machine precision, 24 random pairs). It also corrected up2's wall attribution: the demanded step is 57–14,079 int12 units, multi-coordinate, not a basis limit |
| P4 | **Semantic serialization split (sz1)** | 8,284 B of interleaved fp16 metadata byte-planed before the container's Brotli pass; exact inverse permutation in the receiver; zero-transmitted-byte versioning in a reserved header bit | accounting §7.2, §8.3; `.omx/research/ddm_sz1_semantic_metadata_split_receiver_close_20260818.md` | **NO — DROPPED** at generation 4. `reserved = 0`, `semantic_split = false`. The row-prune changed the semantic body length, so re-measured on the edited body the split is **negative**. Receiver support ships and is inert | was −515/520 B; **520 of the §7.4 cumulative 2,829 B are NOT in these bytes** |
| P5 | **cw1 container/window canonicalization** | Five win-families inventoried from source | `.omx/research/ddm_cw1_win_family_canonicalization_20260819.md`; `.omx/research/ddm_cw1_container_consumer_proof_20260819.json` | apparatus; frontier explicitly UNMOVED | 7,990 lines of 11 arm scripts read at source; F3 → ck2 −657 B + to1 −105 B = **−762 B = −5.074e-04 S** at zero distortion |
| P6 | **rr5 CPR1 lossless rider** | Lossless re-coding rider on the CPR1 basis stream; decode-identity proven with three refusing controls | `.omx/research/ddm_rr5_rider_prestage_20260819.md`; decision in `.omx/research/ddm_pq3_packet_rebase_jg5_20260820.md` | **NO — evaluated and DECLINED** | worth **183 B = ΔS −1.2185e-4**, not the −1.85e-4 the chain budgeted (**66%**) — cross-regime constant transfer one memo downstream. Decisive regardless of size: **lossless ≠ free**; folding changes archive bytes, so the score would be DERIVED, not measured |
| P7 | **ec2 sparse-event HPAC + oriented adapter** | Sparse-event conditioning over the HPAC object | `.omx/research/ddm_ec2_sparse_event_hpac_20260812.md`; `.omx/research/ddm_ec2_oriented_adapter_trainer_20260814.md` | **NO** — `READY_TO_FIRE`, never fired | coordinate payload 413 B Brotli-q11 |

## Class 4 — PORTS AND LOWERINGS

| # | Name | Mechanism | Receipt | In packet | Measured |
|---|---|---|---|---|---|
| L1 | **wc2c split-native HPAC token decoder** | Lifts the `runtime/f26_inflate.py:435-441` hard refusal on `native-hpac` by porting **only the integer model half** (63.8% of per-step time) to C, leaving the 2,121-line stateful float64 corrector chain in Python — the axis with **no FP-reduction-order hazard**. Ships a **forced-scalar twin** as parity control; the built library links libc only | `.omx/research/ddm_wc2c_native_split_identity_and_speedup_20260820.md` | **NO — not in the evaluated tree.** It is the submission critical path and an optional fold on the freeze checklist | **1.774×–1.834×** on the token stage, n600 `[macOS-CPU advisory]` (578.716 → 326.160/315.614 s). Derived PASS bar **1.804× — the measurement STRADDLES it** and is graded undecidable locally. Full-field **bit-exact** reproduction of the jg5 `[contest-CUDA T4]` receipt across all four anchors and the retained 117,964,800-byte token payload |
| L2 | **wc2 wall-clock pass** | Located the wall | `.omx/research/ddm_wc2_wall_clock_pass_20260820.md` | diagnosis | jg5 inflate **1,419.900 s**; token decode **1,341.540 s = 94.5% of it** (95.72% is the share of the 1,401.58 s instrumented-stage sum — different denominator; corrected 2026-08-20 by `ddm_pq8` per `ddm_nv1`) |
| L3 | **F26 CPU unlock — f26p / f26q / f26r** | f26p lifts the MC36 runtime to real CPU capability over one full n600 decode. f26q builds a **fused native lowering** of HPAC + probability construction + the RC64 recurrence — correcting the charter's premise that RC64 was the hot surface (it was already compiled C; the hot surface was Python/Torch causal sparse integer HPAC). f26r adds direct int16 frame context + precomputed conv-A class deltas, with a forced-scalar twin | `.omx/research/ddm_f26p_runtime_cpu_lift_20260814.md`; `.omx/research/ddm_f26q_rc64_native_lowering_20260814.md`; `.omx/research/ddm_f26r_hpac_hot_stage_final_rung_20260814.md` | **NO** — the jg5 tree still carries the `f26_inflate.py:435-441` refusal, and no CPU row exists on these bytes | f26q token stage 383.354 → **203.843 s (1.880632×)**; f26r → **147.005 s (1.386639×)**; forced-scalar twin 146.752 s; derived contest-CPU total 1,321.647 s |
| L4 | **RC64 native lowering (shipped half)** | RC64 backend compiled at decode time | `.omx/research/ddm_f26q_rc64_native_lowering_20260814.md` | **YES** — `runtime/entropy/rc64_backend.c`, compiled by `inflate.sh:32` | — |
| L5 | **wc2 HPAC MPS throughput port** | Imports the hash-pinned reference trainer and changes only device and epoch-admission envelope | `.omx/research/ddm_wc2_hpac_mps_port_20260814.md` | **NO — BLOCKED-ENVIRONMENT**; Torch MPS reported built-but-unavailable | reference profile 2,486.478 s; **~92%** in conv fwd+bwd |
| L6 | **rr2 FreeCorrector native port into `f26_hpac_native.c`** | The projected cure for the wall-clock REFUSE | pointer line; `.omx/research/ddm_wc2_wall_clock_pass_20260820.md` | **NO — OWED, submission critical path** | projected ~331 s — **DERIVED, not measured** |

---

## §5 — Diff against the four published surfaces

Checked row by row against `PR_BODY_DRAFT.md`, `README_PUBLIC.md`, `REPORT_PUBLIC.txt`, and the
gen5 packet's `README.md` / `BORROWED_SUBSTRATE_ACCOUNTING.md`.

**Present and correct in all four:** the ancestry root and all three credited PRs (A1–A3), the
PR #138 and PR #135 priority disclaimers (A7, A8, A10), the joint waterfill (O1), the derived
stop rule (O2), the section-level classes (A4–A6, A12, A16, A17), the residual-payload
withdrawal, the runtime-tree pin, the CPU-axis absence, and the wall-clock WARN.

**Omissions found — the headline count is 11.** All in the same direction: mechanisms of ours
that no public surface named.

| # | Omitted | Class | Lands in |
|---|---|---|---|
| 1 | wc2c split-native token decoder + its bit-exact full-field identity proof (L1) | ORIGINAL port | PR body additional comments; README |
| 2 | The wall-clock diagnosis that **94.5%** of inflate is token decode (L2) — 95.72% is the share of the 1,401.58 s instrumented-stage sum, not of inflate elapsed (corrected 2026-08-20 by `ddm_pq8` per `ddm_nv1`) | ORIGINAL measurement | PR body budget section — the WARN was stated, its CAUSE was not |
| 3 | jg2 tail re-encoder (O4) — the instrument that priced every edit exactly | ORIGINAL | accounting §9.3 sub-rows |
| 4 | Edit-cost superposition law (O5) — why per-chunk rate may be summed | ORIGINAL law | accounting §9.3 |
| 5 | Container transform plane2 / ck2 (O8) — **−657 B, in the shipped ancestry** | ORIGINAL | accounting §9.1 ancestry + §9.3 |
| 6 | Tail-override build step (O9) — **−105 B**, and the reason every prior token win was unreachable | ORIGINAL | accounting §9.3 |
| 7 | Uncapped GN solve (O12), un-interleave discovery (O13), damped GN carrier solve (O14) | ORIGINAL | accounting §9.3 |
| 8 | ma1 within-miss corrector (O11) — **the shipped `free_corrector.py` IS `Ma1WithinMissCorrector`** | ORIGINAL | accounting §9.2 row 7 detail |
| 9 | jg4 checkpoint fix (O15) — the correctness precondition for `delta_trustworthy` | ORIGINAL | accounting §9.3 |
| 10 | Apparatus: seal contract (O16), score arithmetic (O17), stager (O18), census guard (O19), firers (O20), role registry (O21), dual-axis workers (O25) | ORIGINAL apparatus | accounting §9.5 (new) |
| 11 | Level-set witness research line (O26) | ORIGINAL research, **not in packet** | accounting §9.5, marked research-only |

**Over-claims found: 0 in the published surfaces.** Two statements were checked hardest and
both hold: the `ours-original` label on the joint waterfill, and the "0 counted archive bytes"
line. **But four over-claims were caught in this arm's own draft before publication** — see §6.

**Three near-misses corrected in the delta document rather than left standing:**

1. The PR body says the compression entry point "has not been re-run for this candidate". True
   but incomplete: it also **cannot** be. "Not re-run" invites a reader to think a re-run would
   close it.
2. **The two wall-clock memos disagree on the residual band.** pq3 and the report use
   `[890.6, 1430.6] s` and grade WARN; wc2 uses `[822, 1302] s` and reads the same body as
   REFUSE. Both are ours. Publishing one without naming the other would be selective quotation.
   Reconciliation is owed to MAIN before either is quoted as the band.
3. **24 of the 34 files in the jg5 candidate tree have no repo source of any kind** — including
   `runtime/f26_inflate.py`, `runtime/residual_archive.py`, `runtime/free_corrector.py`, all of
   `cpr1/`, and `inflate.py` / `inflate.sh`. **The shipping decoder is not in version control.**
   Measured by wc2c's 34-file census. This bears on the reproducibility section and is routed
   to MAIN, not patched here.

---

## §6 — Five charter premises this arm falsified by recall

Recorded because four of them would have produced an over-claim, and because a charter is not
evidence ([[my_own_charters_fail_the_charter_time_optimal_form_law_20260817]]).

| # | Charter said | Measured |
|---|---|---|
| 1 | The repo bundle "PREDATES the sub-0.15 row … parts are stale" | **False.** `ddm_pq3` rebased all four documents onto jg5 on 2026-08-20 ~02:5xZ. They carry the correct archive sha, byte count and score |
| 2 | The compression script's "rc64 pin was found stale — cure: pin = live runtime member `05839d14`" | **False, and the cure would have been a defect.** `05839d14` is the SHIPPED **decoder-only** body (5,638 B, 237 copies); the pin correctly names the **encoder** role (`5c75e2c7`, 12,222 B). Pinning the decoder would break the encode stage. Receipt: `reverse_engineering/rc64_backend_role_registry.json`. What WAS stale is the script's comment saying **two** bodies wear the filename — the registry measured **four** |
| 3 | "wc2 split-native HPAC port with **NEON** + scalar twin" | **NEON is not receipted anywhere in scope.** What is receipted is the forced-scalar twin parity control and a libc-only build with OpenMP removed. Claiming NEON would be an unreceipted mechanism claim |
| 4 | "joint `{edit, drop, keep}` n600 solve" as a shipped original | **`drop` is NOT shipped.** jg3 declared it a MECHANISM reduction with cause: rc4's drop needs a receiver change the pointer body has no path for. Shipped is `edit` + `keep` |
| 5 | "container transforms plane2" listed alongside the dropped semantic split | **Two different mechanisms.** ck2's plane2 container transform IS in the shipped ancestry (−657 B); sz1's semantic serialization split is DROPPED and inert. Conflating them would have simultaneously omitted a real win and claimed a dropped one |

---

## §4 — The compression script's expressibility boundary

`experiments/ddm_pq2_compress_e2e.py` rebuilds the TOKEN stream (optionally plus a declared
container repack) and carries the other seven sections through verbatim. The jg5 candidate's
chain also re-decides content in sections the script copies — the seg token edit solve, the
edit splice, the admission waterfill, and the carrier re-solve. **No recipe can close that
gap.** The script now refuses those candidates by name with their real builders cited, rather
than answering "pass `--recipe-json`" — true for a missing recipe, false for a missing stage.

## Owed

1. `tools/stage_contest_submission_packet.py` has no tests (O18). Named by pq3, still owed.
2. L6 — the rr2 FreeCorrector native port — is the submission critical path and unbuilt.
3. L1's speedup straddles its own PASS bar and is `[macOS-CPU advisory]`; the shipping axis is
   unmeasured.
4. The two residual-band memos disagree (WARN vs REFUSE). Reconcile before quoting either.
5. 24 of 34 shipping-tree files have no repo source. The shipping decoder is not in VCS.
6. Residual-payload provenance (A16) remains unresolved; the claim stays withdrawn.
7. `drop` (A15) is owed headroom at −0.002929 S, blocked on a receiver path.
8. The true cause of the rr2 T4 refusal (A13) remains OPEN; the chartered explanation was
   falsified and the differential test refused to convict.
