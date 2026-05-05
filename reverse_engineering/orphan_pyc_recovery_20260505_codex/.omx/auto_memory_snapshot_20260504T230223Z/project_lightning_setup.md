---
name: Lightning AI Setup — SSH Key + T4 GPU
description: Lightning SSH working. T4 GPU 15GB. User s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai
type: reference
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Connection (updated 2026-04-12)
```
ssh s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai
```

Key: `~/.ssh/lightning_rsa`
GPU: Tesla T4, 15360 MiB VRAM
Host: ip-10-192-11-241

## Setup token (for reconnect after key rotation)
```
curl -s "https://lightning.ai/setup/ssh?t=dac13c0b-ef09-4d29-b99b-0551f2626713&s=01knw7wnzbe79wfq5mqqbx1mbz" | bash
```
If bash fails, run setup steps manually (download key, set permissions, add SSH config).

## Deploy
Use `src/tac/deploy/lightning/deploy.sh` for self-contained deployment.
Precomputed data at `/home/zeus/content/pact/precomputed/`.

## GPU tiers (monthly free)
- T4: 79 hours (baseline training)
- L40S: 5 hours (~4x faster, 48GB VRAM)
- A100: 3 hours
- H100: 1 hour
