# HiNeRV Official Source-Parity Audit

Schema: `hinerv_official_source_parity_audit.v1`
Authority: `false_authority_source_audit_no_score_claim`
Official repo: `/Volumes/VertigoDataTier/pact/experiments/results/oss_nerv_source_audit_20260602T113720Z/repos/HiNeRV`
Official head SHA: `fdb92ec22492246f800621dfd454f6a5c62ab75b`

## Verdict

- official source markers present: `True`
- local receiver bindings present: `True`
- official forward parity proven: `False`
- official forward parity falsified: `True`
- score claim: `False`

## Marker Groups

| group | present | missing |
|---|---:|---|
| `official_hierarchical_feature_grid` | `True` | `` |
| `official_convnext_decoder` | `True` | `` |
| `official_patch_dataset_path` | `True` | `` |
| `official_quant_prune_torchac_bitstream` | `True` | `` |
| `official_config_family_controls` | `True` | `` |

## Blockers

- `hinerv_official_forward_parity_missing`

## Component States

| component | proven | falsified | classification |
|---|---:|---:|---|
| `core_hierarchical_renderer` | `False` | `True` | `receiver_visible_official_like_renderer_without_source_forward_replay` |
| `patch_dataset_path` | `False` | `True` | `frame_index_receiver_path_without_official_patch_dataset_replay` |
| `prune_quant_codec` | `False` | `True` | `receiver_visible_prune_quantnoise_without_official_torchac_replay` |

## Next Actions

- run tiny official HiNeRV torch forward versus local portable/MLX replay with weight/input/output SHA evidence
