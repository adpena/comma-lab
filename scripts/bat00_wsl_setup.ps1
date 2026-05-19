# bat00_wsl_setup.ps1 — Run this ONCE on bat00 to set up WSL2 sshd
# Right-click → Run with PowerShell (as Administrator)

Write-Host "=== bat00 WSL2 SSH Setup ==="

# 1. Fix Windows OpenSSH rate limiting
Write-Host "`n[1/5] Fixing Windows OpenSSH MaxStartups..."
$sshdConfig = "C:\ProgramData\ssh\sshd_config"
if (Test-Path $sshdConfig) {
    $content = Get-Content $sshdConfig -Raw
    if ($content -notmatch "MaxStartups 100") {
        Add-Content $sshdConfig "`nMaxStartups 100:30:200"
        Restart-Service sshd
        Write-Host "  MaxStartups set to 100:30:200"
    } else {
        Write-Host "  Already configured"
    }
} else {
    Write-Host "  WARNING: sshd_config not found"
}

# 2. Configure WSL2 to not idle-shutdown
Write-Host "`n[2/5] Setting WSL2 idle timeout to infinite..."
$wslconfig = "$env:USERPROFILE\.wslconfig"
if (-not (Test-Path $wslconfig)) {
    @"
[wsl2]
vmIdleTimeout=-1
"@ | Out-File $wslconfig -Encoding ascii
    Write-Host "  Created .wslconfig"
} else {
    Write-Host "  .wslconfig already exists (check vmIdleTimeout manually)"
}

# 3. Install + configure sshd inside WSL2 on port 2222
Write-Host "`n[3/5] Setting up sshd in WSL2 on port 2222..."
wsl -d Ubuntu-24.04 -u root -e bash -c @'
apt-get update -qq && apt-get install -y -qq openssh-server > /dev/null 2>&1
sed -i 's/^#\?Port .*/Port 2222/' /etc/ssh/sshd_config
sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication yes/' /etc/ssh/sshd_config
sed -i 's/^#\?PubkeyAuthentication .*/PubkeyAuthentication yes/' /etc/ssh/sshd_config
service ssh restart
echo "  sshd configured on port 2222"
'@

# 4. Port forward 2222 from Windows to WSL2
Write-Host "`n[4/5] Setting up port forwarding (2222 -> WSL2)..."
$wslIp = (wsl -d Ubuntu-24.04 hostname -I).Trim()
netsh interface portproxy delete v4tov4 listenport=2222 listenaddress=0.0.0.0 2>$null
netsh interface portproxy add v4tov4 listenport=2222 listenaddress=0.0.0.0 connectport=2222 connectaddress=$wslIp
Write-Host "  Forwarding 0.0.0.0:2222 -> ${wslIp}:2222"

# 5. Firewall rule
Write-Host "`n[5/5] Adding firewall rule for port 2222..."
Remove-NetFirewallRule -DisplayName "WSL2 SSH" -ErrorAction SilentlyContinue
New-NetFirewallRule -DisplayName "WSL2 SSH" -Direction Inbound -LocalPort 2222 -Protocol TCP -Action Allow | Out-Null
Write-Host "  Firewall rule added"

# 6. Create a startup task so this survives reboots
Write-Host "`n[Bonus] Creating startup task for WSL2 sshd..."
$action1 = New-ScheduledTaskAction -Execute "wsl.exe" -Argument "-d Ubuntu-24.04 -u root -- service ssh start"
$action2 = New-ScheduledTaskAction -Execute "netsh.exe" -Argument "interface portproxy add v4tov4 listenport=2222 listenaddress=0.0.0.0 connectport=2222 connectaddress=$wslIp"
$trigger = New-ScheduledTaskTrigger -AtLogon
Unregister-ScheduledTask -TaskName "WSL2-SSH-Setup" -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName "WSL2-SSH-Setup" -Action $action1,$action2 -Trigger $trigger -RunLevel Highest | Out-Null
Write-Host "  Startup task created"

Write-Host "`n=== Setup Complete ==="
Write-Host "From Mac: ssh `$env:BAT00_USER@`$env:BAT00_IP -p 2222"
Write-Host "(set BAT00_USER and BAT00_IP env vars to your Tailscale credentials)"
Write-Host "This drops you directly into WSL2 Linux with CUDA access."
