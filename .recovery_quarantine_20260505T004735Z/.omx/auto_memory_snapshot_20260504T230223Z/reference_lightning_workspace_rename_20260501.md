---
name: Lightning workspace renamed comma-lab/lossy-compression-challenge — SSH host unchanged
description: 2026-05-01 ~02:10 UTC. User renamed Lightning workspace to "comma-lab" and project to "lossy-compression-challenge". SSH host (s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai) still works — the SSH binds to the underlying Studio machine, not the workspace/project naming. Studio still in CPU mode (4×CPU); GPU switch in UI required for any contest-CUDA dispatch.
type: reference
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Verified post-rename (2026-05-01 ~02:10 UTC)

- **SSH**: `ssh s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai` returns `ok` and `hostname=ip-10-192-11-142`
- **Path**: `/teamspace/studios/this_studio/` still the canonical Studio root
- **Visible dirs** (from screenshot file browser):
  - `pact/` — current synced repo
  - `pact_pfp16_exa.../` — PFP16 evaluation harvest from earlier session
  - `pact_stale_1777.../` — stale snapshot
  - `tac/` — tac wheel install / source mirror
  - `upstream/` — upstream code
  - `main.py` — placeholder
- **GPU mode**: NOT enabled — Studio shows `4×CPU` selected; "Switch to GPU" button visible but not pressed.
- **Workspace name (UI display only)**: `Alejandro Pena / comma-lab / lossy-compression-challenge` (was: `Alejandro Pena / scratch-studio-devbox` or similar)

## Implications

- Per `feedback_lightning_ai_ssh_credentials_20260430.md` and `scripts/lightning_repro_workspace.py:DEFAULT_REMOTE_PACT="/teamspace/studios/this_studio/pact"`: paths still valid, no script changes needed.
- The `~/.ssh/config` entry `Host scratch-studio-devbox` still works (the host alias resolves to the same SSH endpoint).
- For any contest-CUDA dispatch, user must explicitly click "Switch to GPU" in the Lightning Studio UI and pick L40S/H100 (selecting CPU mode = no GPU = ~30s smoke test only, no real eval).

## Cross-refs

- feedback_lightning_ai_ssh_credentials_20260430.md (the auth/path setup)
- scripts/lightning_repro_workspace.py (canonical Studio sync tool, paths verified post-rename)
- Image evidence: 2026-05-01 ~02:10 UTC screenshot of Lightning UI showing workspace selector + CPU mode
