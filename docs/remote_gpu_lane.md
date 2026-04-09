# Remote GPU Lane

Last refreshed: `2026-04-08 16:43:45 -0500`

## Purpose

This document describes the intended permanent setup for remote GPU lanes such as `bat00`.

The goal is not merely “make SSH work once.” The goal is:
- reproducible remote execution
- low-friction log retrieval
- no shell-history signal loss
- no ambiguity about whether a machine is a real lane or a half-working hack

## Official challenge constraints

Per the upstream challenge README:
- official evaluation has a `30` minute limit
- if the submission does **not** require a GPU for inflation, the evaluator uses `CPU: 4, RAM: 16GB`
- if the submission **does** require a GPU for inflation, the evaluator uses a `T4` instance with `VRAM: 16GB, RAM: 26GB`

Implication:
- `bat00` is useful as a CUDA training/proxy/debugging box
- `bat00` is **not** the final hardware authority for GPU-feasibility, because its `RTX 2070 SUPER` has `8GB` VRAM

## Current observed state

`bat00`:
- host auth works via `adpena@bat00.local`
- WSL path works at `/home/adpena/pact-side`
- `uv` is installed in WSL
- CUDA works in WSL
- the trainer now gets through:
  - device init
  - scorer load
  - decode
  - saliency load
  - baseline
- remaining issue: first-epoch training still does not complete cleanly

`mini`:
- stable proxy/eval side lane
- Python `3.12` + `uv` works

`molt`:
- reachable
- not yet hardened enough to count as active capacity

## Why WSL-on-Windows feels fragile

The current `bat00` shape is:
- SSH into Windows
- run `wsl -e bash -lc ...`
- launch Linux jobs through a Windows shell boundary

That is inherently brittle because:
- quoting has to survive Windows shell parsing and then Linux shell parsing
- process supervision is split across Windows and Linux
- logs can land in places that are awkward to inspect remotely
- environment assumptions differ between host Windows and guest Linux

This is acceptable for bootstrap and recovery work. It is not the ideal permanent shape.

## Permanent target shape

Preferred end state:
1. Enable `systemd` in WSL.
2. Run `openssh-server` **inside WSL**.
3. SSH directly into the Linux environment instead of hopping through Windows shell parsing.
4. Keep a single repo root under WSL with a stable `uv` cache path and stable log path.
5. Launch all remote jobs through a manifest-backed helper, not hand-written shell blobs.

## Recommended setup

### 1. Enable systemd in WSL

Follow Microsoft’s official WSL systemd guidance.

Desired file:

```ini
# /etc/wsl.conf
[boot]
systemd=true
```

Then restart WSL:

```powershell
wsl --shutdown
```

Why:
- `systemd` makes service management predictable
- `sshd` can run as a real Linux service

### 2. Install and enable OpenSSH inside WSL

Inside WSL:

```bash
sudo apt-get update
sudo apt-get install -y openssh-server
sudo systemctl enable ssh
sudo systemctl start ssh
```

Why:
- direct SSH into Linux removes most of the Windows quoting fragility
- remote jobs become normal Linux processes

### 3. Prefer direct WSL SSH networking

Microsoft documents WSL networking modes, including mirrored networking on newer Windows builds.

Preferred:
- mirrored networking or another direct-reachability mode where WSL itself is reachable

Fallback:
- explicit Windows port forwarding to the WSL `sshd`

Why:
- the direct Linux endpoint is what we actually want to manage

### 4. Keep the environment pinned

Inside WSL:
- use one repo root, for example `/home/adpena/pact-side`
- use one cache root, for example `/tmp/uv-cache` or a larger persistent path if `/tmp` is too small

Recommended environment:

```bash
export UV_CACHE_DIR=/tmp/uv-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Why:
- stable cache paths reduce one-off failures
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is explicitly recommended by PyTorch for fragmentation-sensitive CUDA workloads

### 5. Use remote manifests, not shell memory

Every remote job should have:
- host
- remote root
- command
- log path
- timestamped manifest

This repo now has the first helper for that:
- `experiments/remote_job.py`

## Practical lane policy

`bat00` should be used for:
- CUDA training
- CUDA smoke tests
- throughput experiments

`bat00` should **not** be used as:
- promotion authority
- a substitute for the official CPU scorer path

## Sources

- NVIDIA CUDA on WSL User Guide:
  - https://docs.nvidia.com/cuda/wsl-user-guide/
- Microsoft Learn: Use systemd to manage Linux services with WSL:
  - https://learn.microsoft.com/en-us/windows/wsl/systemd
- Microsoft Learn: OpenSSH for Windows overview:
  - https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh-overview
- Microsoft Learn: Advanced settings configuration in WSL:
  - https://learn.microsoft.com/en-us/windows/wsl/wsl-config
- Microsoft Learn: Accessing network applications with WSL:
  - https://learn.microsoft.com/en-us/windows/wsl/networking
