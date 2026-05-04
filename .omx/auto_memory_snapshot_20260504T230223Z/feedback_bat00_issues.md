---
name: bat00 WSL Issues  
description: bat00 is FULLY SET UP but needs physical access to launch training
type: feedback
---

bat00 RTX 2070 Super is FULLY SET UP:
- WSL: DNS fixed, curl, git-lfs, screen installed
- Python: 3.14 venv at ~/tac/.venv with PyTorch+CUDA
- CUDA: confirmed working (torch.cuda.is_available() = True)
- tac library: copied from repo, train_tac.py ready
- Upstream: cloned with LFS (scorer models + GT video)
- Autocast fp16: added to eval loop (fixes 8GB VRAM limitation)
- boot keepalive: wsl.conf [boot] command = /usr/bin/sleep infinity

**BLOCKING ISSUE**: SSH→PowerShell→WSL triple-quoting makes remote
command execution unreliable for complex operations. Screen/nohup
processes die because WSL sessions terminate.

**Fix**: User must physically be at bat00 to launch training:
```
wsl
screen -S tac
cd ~/tac && source .venv/bin/activate
bash run_training.sh
# Ctrl+A, D to detach
```

Screen session will persist. Reattach: `screen -r tac`

**Alternatively**: Install OpenSSH server inside WSL itself
(`sudo apt install openssh-server`) and SSH directly to WSL's IP,
bypassing PowerShell entirely.
