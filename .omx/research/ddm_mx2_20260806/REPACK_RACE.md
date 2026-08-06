# ddm_mx2 Repack Race

| candidate | raw B | best generic | delta generic B | CPR1 | reason |
|---|---:|---:|---:|---|---|
| `move_0023_snap_r00_c12_L13.zip.receipt-bytes:0.bin:bulk` | 341296 | stored 341296 | 0 | NOT_COMPATIBLE | legacy carrier length mismatch: 341296 != 28176 |
| `move_0023_snap_r00_c12_L13.zip.receipt-bytes:0.bin:config` | 36 | stored 36 | 0 | NOT_COMPATIBLE | legacy carrier length mismatch: 36 != 28176 |
| `move_0023_snap_r00_c12_L13.zip.receipt-bytes:0.bin:renderer` | 3266 | stored 3266 | 0 | NOT_COMPATIBLE | legacy carrier length mismatch: 3266 != 28176 |
| `move_0023_snap_r00_c12_L13.zip.receipt-bytes:0.bin:selector` | 535 | brotli 277 | -258 | NOT_COMPATIBLE | legacy carrier length mismatch: 535 != 28176 |
| `move_0023_snap_r00_c12_L13.zip.receipt-bytes:0.bin:pose_warp` | 8751 | stored 8751 | 0 | NOT_COMPATIBLE | legacy carrier length mismatch: 8751 != 28176 |
| `move_0023_snap_r00_c12_L13.zip.receipt-bytes:0.bin:frame0_pose_repair` | 4081 | lzma 4078 | -3 | NOT_COMPATIBLE | legacy carrier length mismatch: 4081 != 28176 |
| `archive.zip:0.bin:bulk` | 341295 | stored 341295 | 0 | NOT_COMPATIBLE | legacy carrier length mismatch: 341295 != 28176 |
| `archive.zip:0.bin:config` | 36 | stored 36 | 0 | NOT_COMPATIBLE | legacy carrier length mismatch: 36 != 28176 |
| `archive.zip:0.bin:renderer` | 3266 | stored 3266 | 0 | NOT_COMPATIBLE | legacy carrier length mismatch: 3266 != 28176 |
| `archive.zip:0.bin:selector` | 535 | brotli 277 | -258 | NOT_COMPATIBLE | legacy carrier length mismatch: 535 != 28176 |
| `archive.zip:0.bin:pose_warp` | 8751 | stored 8751 | 0 | NOT_COMPATIBLE | legacy carrier length mismatch: 8751 != 28176 |
| `archive.zip:0.bin:frame0_pose_repair` | 4312 | stored 4312 | 0 | NOT_COMPATIBLE | legacy carrier length mismatch: 4312 != 28176 |
| `archive.zip:state/renderer.sec` | 3341 | stored 3341 | 0 | NOT_COMPATIBLE | legacy carrier length mismatch: 3341 != 28176 |
| `archive.zip:state/selector.sec` | 535 | brotli 277 | -258 | NOT_COMPATIBLE | legacy carrier length mismatch: 535 != 28176 |
| `archive.zip:state/pose_stub.sec` | 83 | brotli 76 | -7 | NOT_COMPATIBLE | legacy carrier length mismatch: 83 != 28176 |
| `archive.zip:state/pose_warp.stp` | 6864 | stored 6864 | 0 | NOT_COMPATIBLE | legacy carrier length mismatch: 6864 != 28176 |
| `ddm_sc1_20260728:ep_chunks:e_p_float32` | 14400 | brotli 12823 | -1577 | NOT_COMPATIBLE | legacy carrier length mismatch: 14400 != 28176 |

## Summary

- axis: lossless byte-only local repack; no scorer forward
- candidates tested: 17
- CPR1 legacy carrier expected bytes: 28,176
- CPR1 applied: 0
- CPR1 applicable to current banked sections: false
- JSON receipt: `/Volumes/VertigoDataTier/pact/ddm_mx2_20260806/repack_race/repack_race.json`
- JSON sha256: `268e83faa8871c750fc8d61ee60a2572d0384f48bc82e4d2f149f39b97893459`

The generic-code deltas in this table are exact byte-only round-trips of isolated payload bytes. They are not archive-level admissions because the current IX2/PFS1 layouts already include container, coder-state sharing, or member framing choices. The only admitted race verdict for mx2 is the bounded one: CPR1 is not applicable to the currently banked pose-adjacent sections because none are PR130 legacy-carrier-shaped.

The first attempted tq1 parse used four joint section names and failed closed; the actual tq1 IX2 payload has five joint sections including `frame0_pose_repair`. The final run used the five-section map and closed exactly.
