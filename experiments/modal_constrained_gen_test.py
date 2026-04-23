"""Test constrained gen timing on Modal T4 — is it within 30-min budget?"""
import modal

app = modal.App("constrained-gen-timing")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("torch==2.5.1", "torchvision", "safetensors", "einops",
                 "segmentation-models-pytorch", "av", "click", "tqdm")
    .add_local_dir("src/tac", remote_path="/root/tac")
    .add_local_dir("upstream", remote_path="/root/upstream")
)


@app.function(image=image, gpu="T4", timeout=1200)
def test_timing() -> dict:
    import sys, time, torch
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/upstream")
    from tac.constrained_gen import coupled_trajectory_optimize
    from tac.scorer import load_differentiable_scorers

    posenet, segnet = load_differentiable_scorers("/root/upstream", device="cuda")
    masks = torch.randint(0, 5, (20, 384, 512)).long().cuda()
    poses = torch.randn(10, 6).cuda()

    t0 = time.monotonic()
    result = coupled_trajectory_optimize(
        masks=masks, expected_pose=poses,
        posenet=posenet, segnet=segnet,
        num_steps=100, lr=1.0,
        seg_weight=100.0, pose_weight=10.0,
        device="cuda", segnet_loss_mode="hinge",
        eval_roundtrip=True,
    )
    elapsed = time.monotonic() - t0
    per_pair = elapsed / 10
    projected = per_pair * 600 / 60
    print(f"T4: {elapsed:.1f}s for 10 pairs ({per_pair:.1f}s/pair)")
    print(f"Projected 600 pairs: {projected:.0f} min")
    print(f"VIABLE" if projected < 25 else "EXCEEDS budget")
    return {"per_pair_s": per_pair, "projected_min": projected}


@app.local_entrypoint()
def main():
    r = test_timing.remote()
    print(r)
