# C067 Micro Mask Reencode Trust-Region Planner - 2026-05-02

Status: planning-only, no score claim, no GPU dispatch.

## Scope

Implemented `experiments/plan_c067_micro_mask_reencode.py` as a deterministic
JSON planner for tiny C067 AV1 mask-stream reencode candidates. The tool reads
C067-style component traces and/or explicit protected-pair lists, ranks hard
pairs, expands protected decoded-mask frames, assigns protected classes and
regions, and emits local byte-screen candidate configs for target savings bands.

It explicitly refuses broad whole-mask CRF replacement. This preserves the
current forensic recommendation: only micro trust-region reencode probes should
be considered after coarse mask AV1/CRF replacement and broad CMG mutations
showed PoseNet-collapse risk.

## Frontier Anchor

- C067 score: `0.31561703078448233`
- Archive bytes: `276214`
- Archive SHA-256: `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`
- Mask stream: `masks.mkv`, `219472` bytes
- Canonical score source for any future claim: `archive.zip -> inflate.sh -> upstream/evaluate.py` via exact CUDA auth eval

## Sample Planning Screen

Command was run locally into `/tmp` against
`experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/component_trace.json`.
No repo result artifact was created by that sample command.

Planner summary:

- Protected ranked pairs from trace: `76`
- Refused family: broad whole-mask CRF replacement
- Candidate `c067_micro_av1_mask_reencode_save05k`: target savings `5000`, acceptable archive range `270702..271726`, acceptable mask range `213960..214984`, CRF hint `50`, protected pairs `65`, protected mask frames `202`
- Candidate `c067_micro_av1_mask_reencode_save08k`: target savings `8000`, acceptable archive range `267702..268726`, acceptable mask range `210960..211984`, CRF hint `52`, protected pairs `59`, protected mask frames `132`
- Candidate `c067_micro_av1_mask_reencode_save12k`: target savings `12000`, acceptable archive range `263702..264726`, acceptable mask range `206960..207984`, CRF hint `54`, protected pairs `50`, protected mask frames `50`

Top hard-pair signals in the sample trace were pairs `164`, `105`, `106`,
`128`, `69`, `67`, `60`, `64`, `130`, `197`, `136`, and `108`. Default
protected regions are horizon lane band, foveal road center, and ego lower road,
scoped to selected protected mask frames.

## Required Gates

Before any archive from these configs can be treated as more than a local byte
probe:

- The embedded builder policy must be reviewed and, if needed, materialized via
  `--write-policy-jsons`.
- The local builder manifest must remain `score_claim=false` and
  `promotion_eligible=false`.
- Measured archive and mask-stream savings must fall inside the selected byte
  band.
- Protected-region decoded agreement must be exact before any exact eval spend.
- Any remote exact eval requires a dispatch claim first.
