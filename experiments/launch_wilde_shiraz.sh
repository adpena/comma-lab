#!/bin/bash
# Launch WILDE and SHIRAZ on two A100 instances (Vast.ai)
# Estimated: ~21h each, ~$12.60 each, ~$25.20 total
#
# DEPLOYMENT PROCEDURE:
#   1. Search:  vastai search offers 'gpu_name=A100_SXM4 reliability>0.95 inet_down>200 disk_space>30' -o 'dph'
#   2. Create:  vastai create instance <ID> --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel --disk 40
#   3. Wait:    vastai show instances  (wait for 'running')
#   4. Get SSH: vastai ssh-url <INSTANCE_ID>
#   5. Upload:  bash experiments/launch_wilde_shiraz.sh upload <SSH_HOST> <SSH_PORT>
#   6. Launch:  bash experiments/launch_wilde_shiraz.sh wilde <SSH_HOST> <SSH_PORT>
#              bash experiments/launch_wilde_shiraz.sh shiraz <SSH_HOST> <SSH_PORT>
#   7. Monitor: ssh -p <PORT> root@<HOST> 'tail -f /workspace/pact/experiments/results/wilde/train.log'
#   8. Download: scp -P <PORT> root@<HOST>:/workspace/pact/experiments/results/wilde/distill_phase2_best.pt .
set -euo pipefail

COMMAND="${1:-help}"
SSH_HOST="${2:-}"
SSH_PORT="${3:-}"

# Files that must exist on the A100 before training
REQUIRED_DATA=(
    "experiments/results/tto_v7_hinge_500/tto_frames.pt"
    "experiments/results/gt_poses.pt"
    "submissions/robust_current/masks_crf50.mkv"
)

# Common training args (shared between WILDE and SHIRAZ)
# Every arg maps 1:1 to a profile key in profiles.py
COMMON="--tto-frames experiments/results/tto_v7_hinge_500/tto_frames.pt \
    --masks submissions/robust_current/masks_crf50.mkv \
    --gt-poses experiments/results/gt_poses.pt \
    --upstream upstream/ \
    --device cuda \
    --base-ch 32 --mid-ch 48 --motion-hidden 24 --depth 1 \
    --pose-dim 6 --use-dsconv --use-dilation \
    --padding-mode replicate \
    --eval-roundtrip \
    --ema-decay 0.997 \
    --use-per-class-weights \
    --use-swa \
    --use-texture-loss --texture-loss-weight 0.5 \
    --use-linf-penalty --linf-weight 0.01 \
    --use-markov-loss --markov-weight 0.1 \
    --checkpoint-every 100 --eval-every 50 --log-every 25 \
    --seed 42"

case "$COMMAND" in
    upload)
        [ -z "$SSH_HOST" ] && echo "Usage: $0 upload <HOST> <PORT>" && exit 1
        echo "=== Uploading code + data to $SSH_HOST:$SSH_PORT ==="
        SSH="ssh -o StrictHostKeyChecking=no -p $SSH_PORT root@$SSH_HOST"
        SCP="scp -o StrictHostKeyChecking=no -P $SSH_PORT"

        # Create workspace
        $SSH "mkdir -p /workspace/pact"

        # Upload source code
        rsync -avz --progress -e "ssh -o StrictHostKeyChecking=no -p $SSH_PORT" \
            --exclude __pycache__ --exclude "*.pyc" --exclude .git \
            --exclude experiments/results --exclude reports \
            src/ root@$SSH_HOST:/workspace/pact/src/
        rsync -avz --progress -e "ssh -o StrictHostKeyChecking=no -p $SSH_PORT" \
            --exclude __pycache__ --exclude "*.pyc" \
            experiments/train_distill.py experiments/optimize_poses.py \
            experiments/qat_finetune.py experiments/auth_eval_renderer.py \
            experiments/pipeline.py \
            root@$SSH_HOST:/workspace/pact/experiments/

        # Upload upstream (scorer models + evaluate.py)
        rsync -avz --progress -e "ssh -o StrictHostKeyChecking=no -p $SSH_PORT" \
            --exclude __pycache__ \
            upstream/ root@$SSH_HOST:/workspace/pact/upstream/

        # Upload required data files
        for f in "${REQUIRED_DATA[@]}"; do
            echo "Uploading $f..."
            $SSH "mkdir -p /workspace/pact/$(dirname $f)"
            $SCP "$f" "root@$SSH_HOST:/workspace/pact/$f"
        done

        # Install dependencies (C1: av for video decode, C2: ffmpeg for mask decode, C3: pydantic for training config)
        $SSH "apt-get update && apt-get install -y ffmpeg && pip install -q safetensors segmentation-models-pytorch timm brotli einops av pydantic numpy"

        echo "=== Upload complete ==="
        ;;

    wilde)
        [ -z "$SSH_HOST" ] && echo "Usage: $0 wilde <HOST> <PORT>" && exit 1
        echo "=== Launching WILDE on $SSH_HOST:$SSH_PORT ==="
        ssh -o StrictHostKeyChecking=no -p "$SSH_PORT" "root@$SSH_HOST" "
            cd /workspace/pact
            mkdir -p experiments/results/wilde
            export PYTHONPATH=src:upstream:\$PWD
            nohup python3 -u experiments/train_distill.py \
                $COMMON \
                --output-dir experiments/results/wilde \
                --segnet-loss-mode hinge --hinge-margin 1.0 \
                --error-boost 9.0 --error-boost-phase3 49.0 \
                --freeze-motion-phase2 --freeze-renderer-phase3 \
                --pose-weight 10.0 --seg-weight 100.0 --pixel-weight 0.1 \
                --phase1-epochs 600 --phase2-epochs 880 --phase3-epochs 200 \
                --phase1-lr 1e-3 --phase2-lr 3e-4 --phase3-lr 1e-4 \
                --phase1-batch-size 16 --phase2-batch-size 8 --phase3-batch-size 8 \
                > experiments/results/wilde/train.log 2>&1 &
            echo \"WILDE launched: PID=\$!\"
        "
        ;;

    shiraz)
        [ -z "$SSH_HOST" ] && echo "Usage: $0 shiraz <HOST> <PORT>" && exit 1
        echo "=== Launching SHIRAZ on $SSH_HOST:$SSH_PORT ==="
        ssh -o StrictHostKeyChecking=no -p "$SSH_PORT" "root@$SSH_HOST" "
            cd /workspace/pact
            mkdir -p experiments/results/shiraz
            export PYTHONPATH=src:upstream:\$PWD
            nohup python3 -u experiments/train_distill.py \
                $COMMON \
                --output-dir experiments/results/shiraz \
                --loss-mode focal_ste --focal-gamma 2.0 \
                --segnet-loss-mode hinge --hinge-margin 1.0 \
                --error-boost 1.0 \
                --hard-frame-ratio 0.3 --error-replay-every 100 \
                --pose-weight 10.0 --seg-weight 100.0 --pixel-weight 0.1 \
                --phase1-epochs 400 --phase2-epochs 1080 --phase3-epochs 200 \
                --phase1-lr 1e-3 --phase2-lr 3e-4 --phase3-lr 1e-4 \
                --phase1-batch-size 16 --phase2-batch-size 8 --phase3-batch-size 8 \
                > experiments/results/shiraz/train.log 2>&1 &
            echo \"SHIRAZ launched: PID=\$!\"
        "
        ;;

    help|*)
        echo "Usage: $0 <command> <ssh_host> <ssh_port>"
        echo ""
        echo "Commands:"
        echo "  upload  <host> <port>  — Upload code + data to A100 instance"
        echo "  wilde   <host> <port>  — Launch WILDE training"
        echo "  shiraz  <host> <port>  — Launch SHIRAZ training"
        echo ""
        echo "Example:"
        echo "  $0 upload ssh6.vast.ai 21550"
        echo "  $0 wilde  ssh6.vast.ai 21550"
        ;;
esac
