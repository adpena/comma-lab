---
name: bat00 SSH Fragility
description: Windows OpenSSH rate limits and locks up — never send rapid SSH connections, use WSL2 port 2222
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
Windows OpenSSH on bat00 is extremely fragile. MaxStartups rate limiter blocks connections after 3-5 rapid successive SSH commands. Once locked up, sshd may not recover without manual `Restart-Service sshd` from admin PowerShell.

**Why:** Default MaxStartups 10:30:100 + Windows SSH service instability. Our previous approach of sending multiple SSH commands in quick succession (status checks, scp, then exec) reliably triggers the rate limiter.

**How to apply:**
- ALWAYS do everything in a single SSH session when connecting to bat00 port 22
- NEVER send more than 1-2 SSH connections in quick succession to bat00
- Use `scripts/bat00.py` which handles this correctly
- Once WSL2 sshd is set up on port 2222, ALWAYS prefer port 2222 (no rate limiter, direct Linux)
- If bat00 becomes unreachable, the user needs to manually restart sshd on the machine
- The `bat00_wsl_setup.ps1` script needs to be run once as admin on bat00 to enable port 2222
