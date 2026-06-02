# Codex Findings - NeRV OSS and MLX Acceleration Follow-up

axis_scope: [planning/control:false-authority, macOS-local-acceleration:false-authority]
score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false

## Sources Checked

- HiNeRV official repo: https://github.com/hmkx/HiNeRV
  - Relevant controls: Accelerate launch path, patch batch sizing, bitstream replay via `--bitstream`, `--bitstream-q`, and refined pruning/quantization pipeline.
- HiNeRV NeurIPS paper page: https://proceedings.neurips.cc/paper_files/paper/2023/hash/e5dc475c370ff42f2f96dddf8191a40c-Abstract-Conference.html
  - Relevant controls: hierarchical positional encodings, frames plus patches, depth-wise convolution, MLP, interpolation layers, and pruning/quantization pipeline.
- SNeRV official repo: https://github.com/qwertja/SNeRV
  - Relevant controls: `train_snerv.py`, `train_snerv_t.py`, `--enc_strds`, `--dec_strds`, `--fc_dim`, `--emb_size`, `snerv_t`, and HNeRV-derived training stack.
- SNeRV arXiv page: https://arxiv.org/abs/2501.01681
  - Relevant controls: DWT LF/HF decomposition, LF encoding, HF decoder generation, MFU, HFR, and temporal TUB path.
- HNeRV official repo: https://github.com/haochen-rye/HNeRV
  - Relevant controls: `--modelsize`, `--quant_model_bit`, `--quant_embed_bit`, ConvNeXt/pixel-shuffle structure, and pruning/finetune flow.
- SR-NeRV arXiv page: https://arxiv.org/abs/2505.00046
  - Relevant control: low internal representation plus super-resolution reconstruction as an embedding-efficiency enhancer.

## Findings

1. HiNeRV is now locally runnable on the intended acceleration substrate. `mlx==0.31.2`, `mlx-metal==0.31.2`, and `accelerate==1.13.0` are installed in `.venv`; MLX reports Apple M5 Max Metal GPU availability. This removes the infrastructure blocker behind `archive_export_backend_not_mlx`.
2. The fresh MLX-backed HiNeRV four-row archive replay is receiver-closed and false-authority: tiny 134842 B, small 247815 B, base 398074 B, wide 812252 B. The only remaining batch blockers are no contest CPU/CUDA replay and no non-rate scorer replay.
3. SNeRV remains a source-parity problem, not a proven method-negative. The current local carrier is explicitly a forked LF-store/HF-generate adapter. Official SNeRV controls still missing for source parity are MFU, HFR, SNeRV_T/TUB, and native MLX train/export.
4. HNeRV/HiNeRV/SNeRV all expose size or capacity knobs (`--modelsize`, `fc_dim`, `emb_size`, quant bits), but contest usefulness is decided by measured marginal score drop versus archive byte price. These knobs should feed the byte-price controller and not be treated as free model quality controls.
5. SR-NeRV's low-resolution internal representation is higher priority as an enhancer than more explicit LF byte-packing, because the contest scorers downsample to 384x512 before evaluation.

## Implementation Consequences

- HiNeRV next implementation should be a real MLX trainer loop consuming decoder-weight waterfill, PR95-style coder/QAT pressure, and exact archive replay after export.
- SNeRV next implementation should port source-faithful MFU/HFR/TUB controls onto MLX-first, NumPy-portable primitives and then compare against the current forked LF/HF adapter.
- MLX/Metal is authorized for dev velocity and gradient/VJP iteration only. It remains false-authority until paired with byte-closed archive/runtime plus contest CPU/CUDA replay.
