---
name: Lab Infrastructure and Machines
description: All available compute — local, cloud free tiers, remote machines. Updated 2026-04-10.
type: reference
---

**Free compute fleet:**
- **Local Mac** (Apple Silicon MPS): Primary. ~5 min/epoch for h=64 with full 600-pair eval.
- **Colab T4** (free 12h/week): Running h=64 standard. Saves to Google Drive.
- **Modal A10G** ($30/month free): h=96 at ep 607+. Resume-capable.
- **Kaggle T4** (30h/week free): Available, used before for SegNet attack.
- **GCP T4** ($300 free credits): Configured, billing unlinked for safety.
  Re-enable: `gcloud billing projects link personal-mbp-2026 --billing-account=019408-1D0332-BB5DBF`
  Budget alert set at $0.01.
- **Lightning AI** (22 GPU-hours/month free): Not yet activated. Worth trying.

**Other machines:**
- **bat00** (RTX 2070 Super 8GB): Reachable via SSH, PowerShell→WSL. CUDA available.
  Setup cost high (WSL quoting pain). Use for targeted CUDA experiments.
- **Mac Mini / Tertiary**: On network, can run scorers on CPU.

**Cloud deploy scripts:**
- `experiments/modal_h96_v2_deploy.py` — Modal A10G, hardened tac
- `experiments/modal_nuclear_deploy.py` — Modal H100, 5000 epochs
- `experiments/cloud_deploy.py` — Universal: GCP, Lightning, Colab
- `experiments/tac_colab.ipynb` — Self-contained Colab notebook
- `experiments/precompute_for_modal.py` — 7.5GB precomputed tensors

**Precomputed fast path:**
- `tac.data.load_frames(precomputed_dir=...)` skips 10 min decode
- `--precomputed` flag in train_tac.py
