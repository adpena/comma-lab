---
name: Vast.ai API Key and SSH Setup
description: Vast.ai API key location, SSH key registration, instance creation patterns
type: reference
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## API Key
- Stored at `~/.config/vastai/vast_api_key` (set via `vastai set api-key`)
- Key value: starts with `1ab172ec...`

## SSH Key
- Registered at account level via `vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)"`
- Key type: ed25519
- Must be registered BEFORE creating instances (or use `vastai attach ssh <instance_id> "<pubkey>"`)
- Instances created with `--ssh` flag will use the account-level key

## Instance Creation Pattern
```bash
vastai create instance <offer_id> \
  --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime \
  --disk 50 \
  --ssh \
  --onstart-cmd 'bash -c "apt-get update && apt-get install -y git git-lfs ffmpeg unzip && pip install safetensors av einops timm click pydantic segmentation-models-pytorch numpy"' \
  --label "pact-exp-NAME"
```

## Search Pattern
```bash
vastai search offers 'gpu_name=RTX_4090 reliability>0.95 inet_down>200 disk_space>30 num_gpus=1' -o 'dph' --raw
```

## SSH Connection
```bash
ssh -p <port> -o StrictHostKeyChecking=no root@<ssh_host>
```

## Key Lessons
- SSH key must be at account level BEFORE instance creation, otherwise SSH fails with "Permission denied"
- `vastai attach ssh` on existing instances may not work reliably — destroy and recreate instead
- Onstart script runs before SSH is available — wait until `actual_status=running` and SSH connects
- Instances cost money from creation — destroy promptly when done
