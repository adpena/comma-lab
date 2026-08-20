# PR95 MLX-trained byte-closed contest archive

Packaged by `tools/package_pr95_mlx_pytorch_state_dict_to_contest_archive.py`
(canonical loop closure cascade #2 per the PR95 MLX cascade plan).

## Custody

| Field | Value |
| --- | --- |
| `archive_sha256` | `ddbeda1e4832430fd985df16eccfc67a8fb48ed29ec0e0323301b6c4c07d821c` |
| `archive_size_bytes` | `178,417` |
| `archive_member_count` | 1 (member name: `0.bin`) |
| `decoder_state_dict_tensor_count` | 28 |
| `latent_shape` | `[600, 28]` |
| `source_archive_sha256` | `e976acd5fe565c94fb9a8c62e5200c949919f76150e84599f268d6a58588440a` |
| `pytorch_state_dict_sha256` | `f835e816be0c015bc649f4a83b00c6de80841db90f236b4932a22fc2227bbcf3` |
| `upstream_pr_number` | 95 |

## Inflate

```bash
./inflate.sh <archive_dir> <output_dir> <file_list>
```

The runtime tree is self-contained per Catalog #295 + HNeRV parity
discipline lesson 9 (`src/model.py` + `src/codec.py` vendored
alongside `inflate.py`).

## Score authority

This packet is `[macOS-MLX research-signal]` source. The byte-closed
archive itself carries no contest-axis score authority until paired
contest CPU + CUDA auth eval lands per CLAUDE.md "Submission auth
eval - BOTH CPU AND CUDA" non-negotiable + Catalog #192/#317/#341
canonical-routing-markers discipline.

## Cascade NEXT

Loop closure cascade piece #3 = full-frame inflate parity test
(MLX-trained forward vs PyTorch byte-closed archive inflate, on the
contest video, byte-for-byte at the rendered uint8 RGB output).
