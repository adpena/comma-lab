# Lab Fleet Setup — Tailscale + WSL2

Mesh network for all lab machines. Every machine gets a stable 100.x.y.z IP
that survives reboots, sleep/wake, and network changes. Zero port forwarding.

## Fleet

| Machine | OS | GPU | Role |
|---------|-----|-----|------|
| M5 Max MacBook | macOS | MPS 128GB | Primary dev |
| tertiary M1 MacBook Pro | macOS | advisory only | Low-memory CPU-only edge worker |
| Intel Mac Mini | macOS | integrated | Build server / CI |
| bat00 (Windows) | WSL2/Windows | RTX 2070S → 3090 | GPU training |

## Step 1: Create Tailscale Account (once)

1. Go to https://login.tailscale.com/start
2. Sign up with GitHub (uses your @adpena account)
3. This creates your "tailnet" — all devices join this network

## Step 2: Install on Each Mac (2 minutes each)

```bash
# Option A: Homebrew (CLI only, lightweight)
brew install tailscale
sudo tailscaled &
tailscale up

# Option B: App Store (GUI + menu bar icon)
# Search "Tailscale" in App Store, install, sign in
```

After sign-in, each Mac gets a 100.x.y.z IP. Check with:
```bash
tailscale ip -4    # your IP
tailscale status   # all devices
```

## Step 3: Install on bat00 Windows (5 minutes)

### 3a: Tailscale on Windows
1. Download from https://tailscale.com/download/windows
2. Install, sign in with same GitHub account
3. Tailscale runs as a Windows service (auto-starts on boot)
4. Note the IP: open Tailscale tray icon → "My IP"

### 3b: WSL2 Setup (if not already done)
Open PowerShell as Admin:
```powershell
wsl --install -d Ubuntu-22.04
# Reboot if prompted
```

### 3c: SSH Server in WSL2
```bash
# Inside WSL2:
sudo apt update && sudo apt install -y openssh-server
sudo systemctl enable ssh
sudo systemctl start ssh

# Set a password for SSH access:
passwd
```

### 3d: Make WSL2 sshd Survive Reboots
Create a Windows Scheduled Task (PowerShell as Admin):
```powershell
$action = New-ScheduledTaskAction -Execute "wsl.exe" -Argument "-d Ubuntu-22.04 -u root -- service ssh start"
$trigger = New-ScheduledTaskTrigger -AtLogon
Register-ScheduledTask -TaskName "WSL2-SSH" -Action $action -Trigger $trigger -RunLevel Highest
```

### 3e: Forward Tailscale port to WSL2
Tailscale assigns the IP to Windows, but sshd runs in WSL2.
Add this to Windows Task Scheduler (runs at logon alongside WSL2-SSH):
```powershell
# In PowerShell (admin) — one-time setup:
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=22 connectaddress=localhost connectport=22
```

This forwards port 22 on the Tailscale IP to WSL2's sshd.

## Step 4: Configure SSH on Primary Mac

Add to `~/.ssh/config`:
```
Host bat00
    HostName 100.x.y.z    # Replace with bat00's Tailscale IP
    User adpena            # Replace with WSL2 username
    ServerAliveInterval 30
    ServerAliveCountMax 120
    ConnectTimeout 10

Host m1-macbook
    HostName 100.x.y.z    # Replace with M1 MacBook's Tailscale IP
    User adpena

Host mac-mini
    HostName 100.x.y.z    # Replace with Mac Mini's Tailscale IP
    User adpena
```

Then:
```bash
ssh-copy-id bat00      # passwordless auth
ssh bat00              # connect!
```

## Step 5: CUDA in WSL2 (bat00)

NVIDIA GPU passthrough works automatically in WSL2 if the Windows
NVIDIA driver is recent (535+). Inside WSL2:

```bash
# Verify GPU is visible:
nvidia-smi

# Install uv + Python:
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv venv ~/.venv --python 3.12
source ~/.venv/bin/activate

# Install PyTorch with CUDA:
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Verify:
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Should print: NVIDIA GeForce RTX 2070 SUPER (or 3090)
```

## Usage

```bash
# From primary Mac:
ssh bat00                                    # shell
ssh bat00 'nvidia-smi'                       # quick GPU check
rsync -avz ./experiments bat00:~/pact/       # sync files
ssh bat00 'cd ~/pact && nohup scripts/bat00_runner.sh &'  # launch job
ssh bat00 'tail -f ~/pact-logs/latest.log'   # monitor

# VS Code (optional):
# Install "Remote - SSH" extension
# Connect to "bat00" → opens WSL2 environment with GPU access
```

## Troubleshooting

**Can't connect after reboot:**
- Check `tailscale status` on both machines
- On bat00: verify WSL2 is running: `wsl -l -v` in PowerShell
- Restart WSL2 sshd: `wsl -d Ubuntu-22.04 -u root -- service ssh restart`

**GPU not visible in WSL2:**
- Update Windows NVIDIA driver to latest
- Ensure WSL2 (not WSL1): `wsl -l -v` should show VERSION 2
- Do NOT install nvidia-driver inside WSL2 — use the Windows driver

**Tailscale disconnected:**
- On Mac: `tailscale up`
- On Windows: right-click Tailscale tray → Connect
