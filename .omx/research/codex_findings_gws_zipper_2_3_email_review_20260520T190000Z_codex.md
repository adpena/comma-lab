# Codex Findings: GWS Zipper 2 and Zipper 3 Email Review

UTC: 2026-05-20T19:00:00Z
Owner: Codex
Lane: lane_gws_zipper_2_3_email_review_20260520
Scope: Read latest Gmail messages with subjects "Zipper 2" and "Zipper 3"; inspect attachments; review for PR110 relevance, theoretical-floor strategy, implementation design, provenance risk, math/score risk, engineering feasibility, actionable tasks, and current Pact constraints.
Mutation scope: No live PR110 files, corrected draft memos, or upstream submission files edited. Raw custody artifacts stored under ignored `.omx/research/artifacts/`.

## GWS Commands Used

High-level only; no secrets exposed:

- `gws gmail users messages list` with Gmail search queries for `subject:"Zipper 2"` and `subject:"Zipper 3"`.
- `gws gmail users messages get` with `format=full` for each returned message id.
- `gws gmail users messages attachments get` for each ZIP attachment id.

`gws schema gmail.users.messages.* --resolve-refs` was attempted first, but the CLI stack-overflowed; direct Gmail method calls succeeded.

## Email Custody

| Subject | Gmail message id | Date header | From | Attachment | Body |
|---|---:|---|---|---|---|
| Zipper 2 | `19e462dfac47776c` | Wed, 20 May 2026 11:17:46 -0500 | Alejandro Pena `<alejandrod.pena@icloud.com>` | `new_pack.zip` | Empty except CRLF |
| Zipper 3 | `19e46525f83e4eb6` | Wed, 20 May 2026 11:57:30 -0500 | Alejandro Pena `<alejandrod.pena@icloud.com>` | `new_pack_updated.zip` | Empty except CRLF |

Raw custody root:

`.omx/research/artifacts/zipper_2_3_gws_review_20260520T190000Z_codex/`

The root is ignored by `.gitignore:239` (`.omx/research/artifacts/`).

## Artifact Hashes

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `new_pack.zip` | 15176 | `ba5a0c9db6a43761e91a92951796fb672d2682625cd12e1303ce34c37b55680f` |
| `new_pack_updated.zip` | 20394 | `7079cf727d5d016773f12c278a896092449f2d7da35cbc3ab4020fe5a05e12ea` |
| `zipper_2_message_full.json` | n/a | `10b5da6171b8b4031e7520052a68689b9da1cca11ded26080001d5f3bb983fd4` |
| `zipper_2_attachment_new_pack.json` | n/a | `6d537f2d0b9e3f22b5b5584823a27c428c04efdc499a72e1371b767b6eda8f2e` |
| `zipper_3_message_full.json` | n/a | `12aad3ff1b9a652e0dd8b8380a843814ad96ead25681a6576aaf37eb7b02ad9f` |
| `zipper_3_attachment_new_pack_updated.json` | n/a | `2e7ed8fee05371520ffd744ba71deca79c4981346a78128a0743511f9027a7e2` |

## Contents Index

`Zipper 2` contains:

- `new_pack/DESIGN_DIRECTION.md` - 128 lines, SHA `7cd938dd3fdbd0e47afb4dec276958d7484bad206d83700dbf50f8effb2e5982`
- `new_pack/README.md` - 42 lines, SHA `dd25363b9fd75197c72e207f04103e38ceea4ecfd06c2b3c4addcaa4a4f81766`
- `new_pack/EXPERIMENT_QUEUE.json` - 67 lines, SHA `5a22a00c9c069117af3ac325bf00adc3944c2aca12bf938bc6f69853335779f6`
- `new_pack/IMPLEMENTATION_SCAFFOLD/hyperbolic_foveation.py` - 151 lines, SHA `4315d2470097de1d3e93dedfb1112096c4fb303195bf2a2da18030a43a6ad173`
- `new_pack/IMPLEMENTATION_SCAFFOLD/visual_primitives_codec.py` - 195 lines, SHA `de013bf461d1b9ac1c2fdd2d0292b0c39973ddd65d86b9b510e724358d8d4a88`
- `new_pack/IMPLEMENTATION_SCAFFOLD/world_model_codec.py` - 225 lines, SHA `4a2dae2b595dcd2605743e3658b523cf8167c6a3ad4d0e338fa11e8e451774eb`

`Zipper 3` is an updated superset:

- Same `DESIGN_DIRECTION.md` and three `IMPLEMENTATION_SCAFFOLD/*.py` files as `Zipper 2`.
- Updated `README.md` - 46 lines, SHA `fc1b45633b45d0ee2d1ebdce3660a103761555cc3f12b6de234d61880702a2cd`
- Updated `EXPERIMENT_QUEUE.json` - 120 lines, SHA `e429ef0f8ac67fde2f1e99d91c4f7a3e2abbcef07830d55ec27e791de5c6ab72`
- New `UPSTREAM_EVAL_AND_OPENPILOT.md` - 42 lines, SHA `a7d38fc3d89bd5aa0aa83911b784218878d7c35a2fc5d601bc38f7c47b66d674`

`Zipper 3` adds three queue items over `Zipper 2`: `exp_eval_baseline`, `exp_dataset_introspection`, and `exp_master_gradient_smoke`.

## Severity-Ranked Findings

### High: Experiment queue commands are not repo-native and would fail if followed literally

All package queue commands for `scripts/prepare_dataset.py`, `scripts/train_hyperbolic_codec.py`, `scripts/detect_objects.py`, `scripts/train_primitives_codec.py`, `scripts/train_world_model_codec.py`, `scripts/evaluate_world_model_codec.py`, `scripts/analyze_camera.py`, `scripts/estimate_master_gradient.py`, and `submissions/submit_*.sh` reference missing paths in this checkout. The repo-native equivalents are existing Pact surfaces such as `experiments/train_substrate_siren.py`, `experiments/train_substrate_c1_world_model_foveation.py`, `tools/extract_master_gradient.py`, RAFT/foveation tests, SIREN readiness audits, and provider/lane-claim helpers.

Disposition: treat `EXPERIMENT_QUEUE.json` as advisory intent only. Do not dispatch or copy commands directly.

### High: `world_model_codec.py` is not an executable scaffold

Smoke import succeeds, but the core forward path fails when exercised with consistent latent dimensions:

`WORLD_MODEL_FORWARD_FAIL RuntimeError Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [2, 64]`

Root cause: `WorldModelCodec.forward()` computes a vector residual `(B, latent_dim)` and passes it into `ResidualCodec.encode()`, whose first layer is `nn.Conv2d` and expects image-like 4D input. Decode has a related shape mismatch: it feeds `zeros_like(coded_residual)` with residual latent size into a world model expecting the frame-latent size. The included `__main__` demo also mismatches `latent_dim=64` with a dummy encoder returning 3 channels.

Disposition: do not use this file as implementation seed. If pursuing the idea, route through the existing `src/tac/substrates/c1_world_model_foveation/` and `src/tac/substrates/z5_predictive_coding_world_model/` surfaces, which already carry archive grammar, score-aware loss, and probe-disambiguator structure.

### High: Provenance citations are non-resolvable inside the package

The package uses opaque citation tokens such as `501539764972148`, `797659174189052`, and `573927677082790` in bracketed line-reference form, but ships no bibliography, URL map, source index, or exported browser source bundle. Some claims are quickly verifiable from local or official sources, but the package itself cannot support source-faithful review.

Verified locally:

- Contest formula matches `upstream/evaluate.py:90-100` and `upstream/README.md:18-25`.
- Baseline example metrics match `upstream/README.md:86-93`.

Verified online:

- comma.ai's official openpilot 0.11 blog supports the 2B Diffusion Transformer, 2.5M minutes, 50M/100M compressor, and small transformer policy claims.

Not fully verified from the package alone:

- Telescope, DeepSeek visual primitives, RAFT variant, LaPose, IMX298-specific claim, and master-gradient citation lines. These may be true, but the package does not provide enough provenance to cite them in public PR110 or as canonical design authority.

Disposition: any citation imported from this pack needs a real URL/source file before it can enter PR, release, or durable design text.

### Medium: PR110 relevance is indirect; most action has already been superseded by existing Zipper intake

The Zipper 2/3 pack does not include PR110 patch text, archive bytes, or a direct PR110 evidence correction. It is mostly a conceptual follow-up: foveation, primitives, pose priors, world-model coding, baseline eval, dataset introspection, and master-gradient smoke.

Prior local intake already converted the earlier Zipper package into Pact-native gates and completed the immediate no-spend follow-ups:

- `.omx/research/zipper_package_intake_and_execution_plan_20260520T145117Z_codex.md`
- `.omx/research/zipper_source_map_surface_audit_20260520T145523Z_codex.md`
- `.omx/research/zipper_followup_readiness_gates_20260520T150513Z_codex.md`

Disposition: Zipper 2/3 should not reopen live PR110 files. It can update the follow-up queue by adding baseline/environment verification, dataset-introspection, and master-gradient route normalization.

### Medium: Theoretical-floor strategy is directionally relevant but not byte-closed

Good strategic signals:

- Keep exact scorer and hardware as arbiter.
- Prefer score-aware foveation/pose/world-model ideas over pure generic compression.
- Use master-gradient/byte sensitivity to prioritize deterministic corrections.
- Treat heavy openpilot world models as training/search inspiration, not runtime payload.

Risks:

- No archive grammar or inflate contract is provided for any new idea.
- Success criteria like "CPU score lower than baseline PR#110 by 0.001" blur evidence axes. CPU evidence can be useful, but PR110/contest promotion still requires current Pact authority labels and exact CUDA/CPU separation.
- "Theoretical floor" language is not backed by a score decomposition or byte-closed construction in this pack.

Disposition: retain as roadmap signal, not floor authority.

### Medium: Hyperbolic foveation scaffold is illustrative, not the claimed learnable transform

The file imports and runs a basic CPU forward smoke, but:

- `alpha`, `p`, and `center` are plain Python/tensor attributes, not `nn.Parameter`s, so the scaffold does not actually train the warp parameters.
- The documented center is "normalised coordinates (y, x)" but the grid uses `[-1, 1]`; default `(0.0, 0.5)` means mid-height/right-of-center, not top-center/vanishing point in image-normalized `[0,1]` terms.
- The inverse is approximated by negating `alpha`, which can become unstable and is not a true inverse.

Disposition: use existing `tools/audit_hyperbolic_foveation_readiness.py`, `src/tac/foveation_field.py`, `src/tac/lapose_foveation_*`, and foveation tests rather than importing this scaffold.

### Medium: Visual primitives scaffold has runtime/protocol issues

Smoke encode/decode works for a normal box, but a boundary box with coordinates exactly `1.0` round-trips to `0.0` because quantization returns `2**bits` and the bit-packer masks it down to zero:

`BOX_ONE_ROUNDTRIP BoundingBox(x0=0.0, y0=0.0, x1=0.0, y1=0.0, label=255)`

Other concerns:

- No count/header/provenance is encoded with the bitstream.
- Coordinates are not clamped or validated.
- `ConditionalDecoder.forward()` only handles one shared box list through `enumerate([boxes])`, so batch semantics are not real.
- YOLOv5/detector suggestions introduce dependency and weight-custody obligations. If boxes are precomputed, they become charged side-info; if detector runs in inflate, detector weights/runtime must be included and justified.

Disposition: useful as a "primitive side-info" idea only; not production code.

### Low: `UPSTREAM_EVAL_AND_OPENPILOT.md` adds useful context but not a submission path

The score formula and baseline metrics are correct against local upstream files. The openpilot world-model facts are consistent with the official comma.ai 0.11 blog. However, this file is a contextual explainer, not a Pact implementation spec. The immediate value is to reinforce that giant world models belong in training/search, while the contest runtime must remain small and byte-closed.

## Recommended Next Actions

1. Do not edit PR110 or public release surfaces from Zipper 2/3. There is no new direct PR110 correction in these emails.
2. Normalize `Zipper 3` queue into existing Pact tasks:
   - `exp_eval_baseline`: already locally anchored by `upstream/README.md`; only rerun if environment drift is suspected.
   - `exp_dataset_introspection`: convert to a repo-native no-spend artifact using existing video/frame tooling; do not depend on nonexistent `scripts/analyze_camera.py`.
   - `exp_master_gradient_smoke`: route through `tools/extract_master_gradient.py` and existing `master_gradient_anchors.jsonl` custody, not `scripts/estimate_master_gradient.py`.
3. Keep SIREN first-anchor local CPU smoke as the best immediate Zipper-derived frontier artifact, per the existing readiness memo:
   `.venv/bin/python experiments/train_substrate_siren.py --video-path upstream/videos/0.mkv --output-dir experiments/results/siren_smoke_<utc> --epochs 3 --device cpu --smoke --skip-archive-build --skip-auth-eval`
4. Treat world-model/foveation/primitives as already-covered by stronger repo-native surfaces. Reactivate only through archive grammar, score-aware loss, exact runtime, and lane-claim gates.
5. Require real URL/source-backed citations before importing any Zipper 2/3 literature claims into durable public text.

## Exact Artifacts Inspected

- `.omx/research/artifacts/zipper_2_3_gws_review_20260520T190000Z_codex/zipper_2_message_full.json`
- `.omx/research/artifacts/zipper_2_3_gws_review_20260520T190000Z_codex/zipper_3_message_full.json`
- `.omx/research/artifacts/zipper_2_3_gws_review_20260520T190000Z_codex/new_pack.zip`
- `.omx/research/artifacts/zipper_2_3_gws_review_20260520T190000Z_codex/new_pack_updated.zip`
- `.omx/research/artifacts/zipper_2_3_gws_review_20260520T190000Z_codex/zipper_2_extracted/new_pack/*`
- `.omx/research/artifacts/zipper_2_3_gws_review_20260520T190000Z_codex/zipper_3_extracted/new_pack/*`
- `upstream/evaluate.py`
- `upstream/README.md`
- `.omx/research/zipper_package_intake_and_execution_plan_20260520T145117Z_codex.md`
- `.omx/research/zipper_source_map_surface_audit_20260520T145523Z_codex.md`
- `.omx/research/zipper_followup_readiness_gates_20260520T150513Z_codex.md`
- `src/tac/substrates/c1_world_model_foveation/`
- `src/tac/substrates/z5_predictive_coding_world_model/`
- Official comma.ai openpilot 0.11 blog: `https://blog.comma.ai/011release/`
- Official comma2k19 dataset surfaces found via Hugging Face/GitHub search (`https://huggingface.co/datasets/commaai/comma2k19`, `https://github.com/commaai/comma2k19`); used only to check dataset-context plausibility, not as a PR110 source.
