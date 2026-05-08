# Representation Integration Gap Audit - 2026-05-08

Owner: codex
Scope: adversarial integration-failure audit for HNeRV/MNeRV/NeRV/Cool-Chic-style representation work that existed in this repo before public HNeRV-family PRs reached the active exact frontier. This is not a strategy essay and does not introduce a new score claim.

Score-claim boundary:

- This ledger only cites score-bearing artifacts that already exist in logs, scorecards, or dated ledgers. It does not recompute, promote, or supersede any score.
- Public rounded values, README claims, CPU proxy bytes, local smoke losses, and planner byte forecasts remain non-promotable unless an existing cited artifact says otherwise.
- I found no literal `MNeRV`/`mnerv` symbol in the inspected source tree. This audit treats MNeRV as a family label adjacent to NeRV/HNeRV-style implicit neural representations, not as a proven named implementation in this checkout.

## Evidence Read

Durable constraints:

- `AGENTS.md:672-700`: packed renderer formats such as `CCh1` and `C3R1` must delegate loader logic, and byte closure alone is insufficient for transplant candidates.
- `AGENTS.md:1048-1076`: high-upside learned-codec lanes must not be retired by absence of clearance, but every block needs an owner, unblock action, experiment, exact negative, or impossibility proof.
- `AGENTS.md:1080-1107`: stack experiments wait on component archives with exact evidence; proxy composition is not enough.
- `AGENTS.md:1205-1218`: Lane 12/Alpha NeRV retraining is build-only until L2 clearance, decoded-baseline masks, valid alpha geometry contract, override forwarding, and non-no-op provenance are satisfied.

Internal representation paths:

- `src/tac/nerv_mask_codec.py:1-30`, `src/tac/nerv_mask_codec.py:66-88`, `src/tac/nerv_mask_codec.py:163-229`: Lane 12 NeRV mask codec exists as a deterministic codec/scaffold with NRV2 scale-table work, but the module explicitly started as pure CPU codec work with training and exact CUDA archive scoring out of scope.
- `experiments/train_nerv_mask.py:1-38`, `experiments/train_nerv_mask.py:77-184`, `experiments/train_nerv_mask.py:202-260`: the NeRV trainer emits `masks.nrv` plus metrics/provenance and can extract SegNet masks, but exact CUDA archive validation remains a separate step.
- `scripts/remote_lane_nerv.sh:1-5`, `scripts/remote_lane_nerv.sh:31-38`, `scripts/remote_lane_nerv.sh:151-194`, `scripts/remote_lane_nerv.sh:222-260`: remote NeRV dispatch was deliberately blocked behind L2 clearance, decoded-baseline mask policy, alpha primitive contract checks, and direct-SegNet forensic flags.
- `scripts/remote_lane_12_owv3_0120_nerv_stack.sh:1-19`, `scripts/remote_lane_12_owv3_0120_nerv_stack.sh:190-214`: NeRV mask replacement was tied to OWv3/pose regeneration because mask changes can invalidate pose assumptions.
- `src/tac/contrib/coolchic_renderer.py:1-10`, `src/tac/contrib/coolchic_renderer.py:42-49`, `src/tac/contrib/coolchic_renderer.py:114-179`: Cool-Chic/C3 code exists, but `CoolChicLatentRenderer` is explicitly not a literal Cool-Chic bitstream and C3 is an experimental residual renderer.
- `src/tac/experiments/train_renderer.py:92-123`, `src/tac/experiments/train_renderer.py:155-160`, `src/tac/experiments/train_renderer.py:2099-2122`, `src/tac/experiments/train_renderer.py:2316-2358`: Cool-Chic/C3 are trainable variants, but they are non-FP4A export variants and the trainer blocks `--auth-eval-on-best` for variants that lack full archive/export support.
- `src/tac/profiles.py:1058-1129`, `src/tac/profiles.py:2846-2877`, `src/tac/profiles.py:4153-4158`: profiles exist for Cool-Chic/C3 and Lane 12 NeRV, but they are smoke/full profile definitions and planning bands, not proof of closed contest packets.
- `reports/local_smoke_coolchic_c3_20260425.md:1-6`, `reports/local_smoke_coolchic_c3_20260425.md:50-61`: local smoke only validated wiring/reproducibility, not quality or byte stability.
- `reports/local_trend_coolchic_c3_20260425.md:1-6`, `reports/local_trend_coolchic_c3_20260425.md:42-74`: C3 float improvement did not automatically survive FP4/export constraints; next gate was mixed precision, CUDA/T4, then archive/inflate.
- `reports/lane_12_nerv_real_archive.json`: NeRV real-archive evidence is CPU/partial and empirical, with mask disagreement and full CUDA training still required.

Public HNeRV-family intake and replay artifacts:

- `experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source/submissions/hnerv_ft_microcodec/src/codec.py:1-9`, `experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source/submissions/hnerv_ft_microcodec/src/codec.py:467-470`: PR101 uses a fixed monolithic payload schema with decoder, latent, and sidecar sections parsed from the archive bytes.
- `experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source/submissions/hnerv_ft_microcodec/inflate.py:19-65`: PR101 inflate consumes the scored archive payload and decodes the model/latents into masks.
- `experiments/results/public_pr103_intake_20260504_codex/source/submissions/hnerv_lc_ac/inflate.py:14-63`, `experiments/results/public_pr103_intake_20260504_codex/source/submissions/hnerv_lc_ac/inflate.py:110-176`, `experiments/results/public_pr103_intake_20260504_codex/source/submissions/hnerv_lc_ac/inflate.sh:11-19`: PR103 keeps a single `x` payload and moves to explicit arithmetic/range-coded sections consumed by inflate.
- `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source/submissions/belt_and_suspenders/src/codec.py:1-14`, `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source/submissions/belt_and_suspenders/src/codec.py:61-139`, `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source/submissions/belt_and_suspenders/src/codec.py:168-254`: PR106 has a fixed HNeRV decoder/latent codec with packed decoder and latent streams.
- `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source/submissions/belt_and_suspenders/inflate.py:27-62`, `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source/submissions/belt_and_suspenders/inflate.sh:17-27`: PR106 original runtime required CUDA and depended on `brotli`; the initial replay failed before score because that dependency was missing from the runtime environment.
- `experiments/results/lightning_batch/exact_eval_public_pr101_hnerv_ft_microcodec_t4_20260504T1302Z/auth_eval.log`, `experiments/results/lightning_batch/exact_eval_public_pr103_hnerv_lc_ac_t4_20260504T1327Z/auth_eval.log`, `experiments/results/lightning_batch/exact_eval_public_pr106_belt_and_suspenders_t4_20260504T1315Z/auth_eval.log`, `experiments/results/lightning_batch/exact_eval_public_pr106_belt_and_suspenders_adapter_t4_20260504T1330Z/contest_auth_eval.adjudicated.json`: existing exact replay/adjudication evidence for PR101, PR103, PR106 failure-before-score, and PR106 adapter success.
- `experiments/results/public_hnerv_frontier_payload_profiles_20260504_codex/scorecard.md:1-10`, `experiments/results/public_hnerv_frontier_payload_profiles_20260504_codex/scorecard.md:18-31`, `experiments/results/public_hnerv_frontier_payload_profiles_20260504_codex/scorecard.md:62-89`: public HNeRV payload scorecard and byte-anatomy routing. The scorecard is the cited source for exact public replay comparisons in this audit.

Roadmap and stack-readiness context:

- `.omx/research/public_hnerv_frontier_deconstruction_20260504_codex.md:23-40`, `.omx/research/public_hnerv_frontier_deconstruction_20260504_codex.md:42-75`, `.omx/research/public_hnerv_frontier_deconstruction_20260504_codex.md:207-243`: public HNeRV exact rows, runtime failure classification, byte anatomy, and stack implications.
- `.omx/research/public_hnerv_adapter_replays_20260504_codex.md:24-52`, `.omx/research/public_hnerv_adapter_replays_20260504_codex.md:54-85`: adapter work classified wrapper/dependency failures separately from method evidence.
- `.omx/research/hnerv_payload_scorecard_followup_20260505_codex.md:5-38`: follow-up scorecard guardrails and no-score safety boundary.
- `.omx/research/frontier_roadmap_status_20260506_codex.md:22-51`, `.omx/research/frontier_roadmap_status_20260506_codex.md:60-74`: acceptance gates for exact-stage candidates and frontier table semantics.
- `.omx/research/roadmap_state_reconciliation_20260508_codex.md:7-25`, `.omx/research/roadmap_state_reconciliation_20260508_codex.md:39-77`, `.omx/research/roadmap_state_reconciliation_20260508_codex.md:89-102`, `.omx/research/roadmap_state_reconciliation_20260508_codex.md:133-167`: current anchored frontier, unanchored roadmap claims, Tier-A readiness gaps, and near-term ordering.
- `.omx/research/cross_paradigm_dispatch_readiness_review_20260508_worker_d.md:73-87`, `.omx/research/cross_paradigm_dispatch_readiness_review_20260508_worker_d.md:94-147`: monolithic bridge, invalid proxy traps, and next-gate checklist.
- `.omx/research/hstack_vstack_hyperprior_repair_20260507_worker_h.md:43-63`, `.omx/research/hstack_vstack_hyperprior_repair_20260507_worker_h.md:65-99`, `.omx/research/hstack_vstack_hyperprior_repair_20260507_worker_h.md:100-108`: HStack/VStack semantic correction, monolithic-section correction, learned-hyperprior gate, and fail-closed policy.
- `src/tac/cross_paradigm_wiring.py:16-20`, `src/tac/cross_paradigm_wiring.py:193-216`: `nerv`, `wavelet`, `vqvae`, and `grayscale_lut` were registered as Alpha stub mask codecs but not wired.
- `tools/pr101_cross_paradigm_hstack_vstack_empirical.py:1-88`, `tools/pr101_cross_paradigm_hstack_vstack_empirical.py:64-75`, `tools/pr101_cross_paradigm_hstack_vstack_empirical.py:191-196`: cross-paradigm stack artifact is CPU byte-anchor/proxy work, with no `inflate.py`, missing side info/latent/sidecar path, and no dispatch readiness.
- `src/tac/tests/test_pr101_cross_paradigm_hstack_vstack.py:171-193`: tests intentionally pin byte-proxy retraction and dispatch blocker behavior.
- `reports/raw/pr101_cross_paradigm_hstack_vstack_corrected_20260508/manifest.json:1-53`, `reports/raw/pr101_cross_paradigm_hstack_vstack_corrected_20260508/manifest.json:198-239`: corrected stack rows still declare `score_claim=false`, `ready_for_exact_eval_dispatch=false`, and byte-proxy/no-real-archive dispatch blockers.

## Concrete Failure Patterns

1. Representation idea existed at the wrong layer.

   Internal NeRV work was a mask codec and training harness, not a closed contest packet. Internal Cool-Chic/C3 work was a renderer research path, not an exportable archive/runtime path. Public PR101/103/106 were not just "HNeRV ideas"; they were packetized codecs with parsers, fixed payload sections, runtime consumers, and exact replay artifacts. The repo had important representation ingredients, but they were not promoted through the same archive grammar and inflate-contract surface.

2. Proxy artifacts were allowed to look like candidates.

   The NeRV path could emit `masks.nrv`; Cool-Chic/C3 could emit local smoke/trend measurements; cross-paradigm/HStack/VStack scripts could emit lower byte proxies. Those outputs did not automatically become charged bytes inside `archive.zip`, nor did `inflate.sh` necessarily consume them. The failure is not lack of creativity; it is letting non-consumed bytes, local losses, or planning rows occupy the mental slot reserved for contest packets.

3. Exportability was discovered after training, not before dispatch design.

   `train_renderer.py` explicitly treats Cool-Chic/C3 as non-FP4A export variants and blocks post-training auth eval when export support is missing. That is the right fail-closed behavior, but it also means the integration plan was backwards: train first, then discover that the representation cannot enter the canonical archive path. Public HNeRV packets had their model shape, payload grammar, and runtime decode path fixed enough for exact replay.

4. Runtime closure was a late adapter problem.

   Public PR106 failed the original replay due to a missing `brotli` dependency before adapter work closed the runtime. Earlier public adapter work also separated wrapper/signature failures from method evidence. This same class likely delayed internal learned-codec lanes: the scored bytes may be promising, but missing import closure, wrong command signatures, absent venv setup, or CUDA assumptions prevent the scorer from reaching the method.

5. Byte layout was modeled as ZIP-member accounting when the frontier used monolithic payload sections.

   PR101, PR103, and PR106 use single charged payloads with internal parser sections. HStack/VStack and cross-paradigm planning had to be corrected away from treating decoder/latent/sidecar as independently swappable ZIP members. Until a candidate proves offsets, lengths, section SHA-256s, decoder consumption, and old/new archive SHA boundaries, its byte accounting is forensic, not a submission path.

6. Component coupling was under-owned.

   NeRV mask replacement is not just a smaller mask stream. Scripts correctly warn that pose regeneration and geometry gates are required when masks change. Public PR106 exact artifacts and deconstruction ledgers show that component balance, especially pose, matters to frontier status. Internal representation experiments often optimized one stream without owning the pose/mask/runtime coupling all the way to exact CUDA.

7. Clearance became a practical sink.

   Lane 12 blockers were valid: decoded-baseline source policy, alpha primitive contract, L2 clearance, non-no-op provenance, geometry gates. But AGENTS now requires blocked high-upside learned codecs to have a named owner and unblock action rather than quietly aging out. A valid block should leave a terminal artifact, exact negative, or next experiment; otherwise it behaves like silent retirement.

8. Public deconstruction became stronger than internal integration.

   After PR101/103/106 intake, the repo gained byte anatomy, scorecards, adapters, and low-level repack routes. That deconstruction discipline was stronger than the pre-public internal integration discipline. The lesson is not to stop deconstructing public packets; it is to impose the same parser/runtime/exact-eval custody on internal representation ideas before they need rescue by external examples.

## Exact-Eval, Archive, Runtime, And Byte-Layout Gaps

Promotion gap checklist observed in the internal paths:

- No canonical `archive_builder` for several representation variants that emits a scored `archive.zip` plus manifest.
- No mandatory proof that the changed representation bytes are inside the scored archive member or parser section.
- No mandatory old/new charged-byte SHA-256 boundary for representation changes.
- No required no-op detector showing that the intended payload changed and that the runtime consumed the changed payload.
- No universal runtime-tree manifest tying `inflate.sh`, Python modules, native dependencies, and package closure to the archive SHA.
- No preflight that fails a learned-codec candidate before training when the chosen variant cannot export into the contest packet format.
- No mask/pose geometry gate that is automatically triggered by a mask representation replacement.
- No parser-section grammar gate for monolithic public-style HNeRV payloads before stack or entropy-code planning is considered score-affecting.
- No strict distinction in some roadmaps between "byte proxy improved" and "exact CUDA candidate got better."

The public HNeRV-family packets passed a different class of test: archive bytes had a fixed schema, inflate parsed the schema, runtime decoded masks, exact replay/adapters reached scorer execution, and scorecards could profile section-level byte mass. Internal paths often stopped one layer earlier.

## Impact On Paradigm, Cross-Paradigm, HStack, And VStack Work

The same pattern is already visible outside HNeRV/NeRV/Cool-Chic:

- Paradigm and cross-paradigm stubs can register a representation without wiring it into a consumed runtime path. `cross_paradigm_wiring.py` explicitly marks several mask codecs as registered but not wired.
- HStack/VStack byte wins can be real planning signal while still failing the contest-packet contract. The corrected manifest keeps `score_claim=false` and `ready_for_exact_eval_dispatch=false` because the artifact is still a CPU byte proxy with no real archive/runtime.
- Rel-error and byte-rate proxies can invert the decision order. A candidate that looks attractive in a manifest but lacks side info, latent blob accounting, K tables, decoder overhead, or runtime consumption should not outrank a less exciting candidate that already has exact archive closure.
- Monolithic-section ignorance can create fake stack opportunities. If a proposed HStack/VStack replacement does not identify the exact PR101/103/106 internal section it replaces, its decoder contract, and the tail-bit/padding/sidecar rules, it is not a stack candidate yet.
- Full learned hyperprior work risks repeating Cool-Chic/C3: a useful float/proxy model can fail promotion if model overhead, quantized side information, deterministic export, and inflate-side decoding are not designed at the same time.

## Prevent-Recurrence Gates

Every learned representation lane should pass these gates before it is called an exact-eval candidate.

1. Representation promotion card

   Required fields: `representation_name`, `target_modes`, `source_artifact`, `archive_builder`, `inflate_consumer`, `runtime_manifest`, `changed_payload_paths`, `old_new_sha256s`, `component_risk_plan`, `exact_eval_command`, `owner`, and `next_unblock_action`.

2. No naked bytes

   If bytes are not inside a scored archive member or parser section and consumed by `inflate.sh`, the artifact must set `score_claim=false` and `ready_for_exact_eval_dispatch=false`. It can be valuable empirical or forensic evidence, but not a candidate.

3. Parser-section gate

   For monolithic HNeRV-family payloads, require a parser manifest with offsets, lengths, section names, section SHA-256s, entropy estimates, and old/new section boundaries. ZIP-member budget rows are invalid unless the packet really has separate ZIP members.

4. Export-first gate

   A trainable renderer or implicit representation must declare its export format before long training. If the variant is non-FP4A, non-int4, or otherwise outside the current archive exporter, the run is research-only until a packet exporter exists.

5. Runtime closure gate

   Run the exact contest inflate signature in a clean environment before dispatch or before treating a public packet as method evidence. Dependency closure failures such as missing `brotli`, wrong wrapper signatures, hidden sidecars, local paths, or CPU/CUDA mismatches must be classified as runtime blockers, not method negatives.

6. Mask/pose coupling gate

   Any mask representation replacement must record decoded mask SHA-256s, mask disagreement, pose-regeneration status, geometry diagnostics, and the exact component-risk plan. Smaller mask bytes alone are insufficient.

7. No-op and provenance gate

   Every byte-level experiment must prove the targeted payload changed, prove the scored runtime consumed it, and preserve old/new archive SHA-256s. Reuse, decode/re-encode, provenance-only changes, and cosmetic ZIP repacks stay forensic until this proof exists.

8. Exact-evidence gate

   Promotion to frontier status requires an existing exact CUDA full-sample artifact with archive bytes, archive SHA-256, runtime tree, command, hardware, sample count, component fields, formula recomputation, logs, and dispatch-claim status. Anything else must carry its lower evidence grade in the artifact itself.

9. Blocker ownership gate

   If a high-upside learned-codec lane is blocked, record one of: active owner and unblock experiment, exact negative and reactivation criteria, compliance impossibility proof, or terminal retirement note. "Waiting for clearance" is not enough.

10. Stack promotion gate

    HStack/VStack/cross-paradigm work must include the real archive boundary, side information, latent streams, K/scale tables, decoder overhead, runtime consumer, and exact-eval plan before it is scheduled as a CUDA candidate. Byte-proxy manifests should continue to exist, but their dispatch fields must remain fail-closed.

## Immediate Audit Conclusions

- The repo had NeRV and Cool-Chic/C3 representation research before the public HNeRV PRs, but most of it lived below the contest packet layer. Public frontier movement came from representations that were already bound to archive grammars and inflate runtimes.
- The integration failure was not "we lacked HNeRV-like ideas." It was "we did not force every promising representation through archive builder, parser, runtime consumer, export, and exact-eval custody early enough."
- The same failure mode can still affect paradigm/cross-paradigm/HStack/VStack work if byte proxies or planning rows are allowed to outrank byte-closed packets.
- The fix is procedural and testable: no naked bytes, parser-section manifests for monolithic packets, export-first design, runtime closure before method judgment, mask/pose coupling gates, and explicit blocker ownership.
