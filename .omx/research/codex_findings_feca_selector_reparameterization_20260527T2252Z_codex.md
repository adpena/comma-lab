# FECa Selector Reparameterization Finding

Generated UTC: 2026-05-27T22:52Z

## Finding

Current CPU-frontier FP11/FECa selector stream can be reparameterized from
`_BlendContextModel.SCALE = 1 << 14` to `SCALE = 256` with `ALPHA_DEFAULT = 2`.
Decoded selector codes are unchanged, source payload bytes are unchanged, and
DQS1 tail bytes are unchanged.

## Artifact

- Source archive: `experiments/results/v14_v2_dqs1_plus_fec10_substituted_20260526T023000Z/submission_dir/archive.zip`
- Source SHA-256: `0a3abfe645c4fac0df9ea89237f25dd9bfc6b2471b897c36d7437795d27d1403`
- Candidate archive: `/Volumes/VertigoDataTier/experiments/results/feca_selector_reparameterized_scale256_20260527Tlocal/submission_dir/archive.zip`
- Candidate SHA-256: `a5f0b1bc148370cc97139cc3d878d2ea9baee980660a9c542f3e4963ff75f4d1`
- Source bytes: `178546`
- Candidate bytes: `178541`
- Saved bytes: `5`
- Selector bytes: `236 -> 231`
- Expected pure-rate score delta: `-0.000003329294765610857` before auth anchoring.

## Proof

- Runtime-consumption proof: `/Volumes/VertigoDataTier/experiments/results/feca_selector_reparameterized_scale256_20260527Tlocal/feca_selector_runtime_consumption_proof.json`
- Full-frame inflate parity proof: `/Volumes/VertigoDataTier/experiments/results/feca_selector_reparameterized_scale256_20260527Tlocal/feca_selector_full_frame_inflate_parity_proof.json`
- Full-frame parity: `true`
- Contest full-sample parity: `true`
- Raw output SHA-256: `00b479229c97ede3e776846297269f7785285702b8dbf3e5dccc733557da605a`
- Raw output bytes: `3662409600`

## Wiring

- Materializer target kind: `selector_stream_context_recode_v1`
- Materializer id: `feca_selector_reparameterize_adapter`
- Unit kind: `selector_stream`
- Operation family: `selector_context_recode`
- Entropy position: `P11 selector_stream`
- Queue resolution: executable with no blockers.
- Materializer observation: `saved_bytes=5`, `rate_positive=true`, `receiver_contract_satisfied=true`.

## Auth Anchor

Modal CPU exact auth eval is dispatched and pending.

- Lane id: `feca_selector_reparam_scale256_cpu_exact`
- Instance/job id: `modal_cpu_feca_selector_scale256_20260527T2247Z`
- Modal call id: `fc-01KSNT0ESDEN4Z5H3S0TDVDKFE`
- Output dir: `experiments/results/modal_auth_eval_cpu/feca_selector_reparameterized_scale256_20260527T2247Z_cpu`

No score claim or promotion claim is made until the auth-eval artifact is
harvested and adjudicated.
