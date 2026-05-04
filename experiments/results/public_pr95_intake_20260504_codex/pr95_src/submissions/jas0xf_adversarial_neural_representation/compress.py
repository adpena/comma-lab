#!/usr/bin/env python
"""Compress: orchestrate the full training + archive pipeline for jas0xf_adversarial_neural_representation.

Pipeline (all stages run sequentially):
  1. master_pretrain   — training/master.py pretrain      → v62_full_best.pt
  2. master_dali_ft    — training/master.py dali-ft       → v62_full_dali_swa.pt
  3. slave_init        — training/slave.py init           → slave_v62_shrink_ultra_swa.pt
  4. slave_ft          — training/slave.py ft             → slave_v62_shrink_ultra_ft_best.pt
  5. slave_dali_ft     — training/slave.py dali-ft        → slave_v62_shrink_dali_best.pt
  6. hpac_train        — training/hpac.py train           → hpac.pt
  7. hpac_ppmd         — (inline) PPMd-compress hpac.pt    → archive_build/hpac.pt.ppmd
  8. archive_build     — training/archive.py              → archive.zip

Total runtime: ~50-60 hours on a single Tesla T4, or ~20-25 hours on RTX 4090.

Use --skip-stage <name> (repeatable) to resume after partial completion.

GPU is required throughout. See training/README.md for the full recipe.
"""
from __future__ import annotations
import sys, io, time, argparse, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN = HERE / "training"
WORK = HERE / "training_workspace"
ARCH_DIR = HERE / "archive_build"

STAGES = [
    "master_pretrain",
    "master_dali_ft",
    "slave_init",
    "slave_ft",
    "slave_dali_ft",
    "hpac_train",
    "hpac_ppmd",
    "archive_build",
]


def run_script(name: str, script: str, args=()):
    if name in SKIP:
        print(f"\n[skip] stage={name}\n", flush=True)
        return
    print(f"\n=== stage={name} script={script} args={list(args)} ===", flush=True)
    t0 = time.time()
    cmd = [sys.executable, "-u", str(TRAIN / script), *args]
    subprocess.check_call(cmd)
    print(f"=== stage={name} done in {(time.time() - t0) / 60:.1f} min ===\n", flush=True)


def stage_hpac_ppmd(hpac_pt: Path, out_ppmd: Path):
    if "hpac_ppmd" in SKIP:
        print(f"\n[skip] stage=hpac_ppmd\n", flush=True)
        return
    print(f"\n=== stage=hpac_ppmd: PPMd-compress {hpac_pt.name} ===", flush=True)
    import torch, pyppmd
    sd = torch.load(hpac_pt, map_location="cpu", weights_only=False)
    buf = io.BytesIO()
    torch.save(sd, buf)
    raw_sz = len(buf.getvalue())
    blob = pyppmd.compress(buf.getvalue(), max_order=4, mem_size=16 << 20)
    out_ppmd.parent.mkdir(parents=True, exist_ok=True)
    out_ppmd.write_bytes(blob)
    print(f"  hpac.pt: {raw_sz:_} bytes -> hpac.pt.ppmd: {len(blob):_} bytes "
          f"({len(blob) / raw_sz:.2%}) -> {out_ppmd}\n", flush=True)


def main():
    global SKIP
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-stage", action="append", default=[],
                    help=f"Stages to skip. Repeatable. One of: {','.join(STAGES)}")
    ap.add_argument("--out-archive", type=Path, default=HERE / "archive.zip",
                    help="Path to write archive.zip (default: <submission_dir>/archive.zip)")
    args = ap.parse_args()

    bad = [s for s in args.skip_stage if s not in STAGES]
    if bad:
        print(f"ERROR: unknown stage(s): {bad}. Valid: {STAGES}", file=sys.stderr)
        sys.exit(2)
    SKIP = set(args.skip_stage)

    WORK.mkdir(parents=True, exist_ok=True)
    ARCH_DIR.mkdir(parents=True, exist_ok=True)

    hpac_pt = WORK / "hpac.pt"
    hpac_ppmd = ARCH_DIR / "hpac.pt.ppmd"

    # Stage 1: Master pretrain (~25 hr T4) — produces v62_full_best.pt
    run_script("master_pretrain", "master.py", ["pretrain"])

    # Stage 2: Master DALI fine-tune (~3.5 hr T4) — produces v62_full_dali_swa.pt
    run_script("master_dali_ft", "master.py", ["dali-ft"])

    # Stage 3: Slave init with ultra-shrunk arch (~7 hr T4)
    #          -> slave_v62_shrink_ultra_swa.pt
    run_script("slave_init", "slave.py",
               ["init",
                "--tag", "ultra",
                "--channels", "24,16,12,8,8,6,6",
                "--d-lat", "6"])

    # Stage 4: Slave fine-tune (~5 hr T4) -> slave_v62_shrink_ultra_ft_best.pt
    run_script("slave_ft", "slave.py", ["ft"])

    # Stage 5: Slave DALI fine-tune (~4.5 hr T4) -> slave_v62_shrink_dali_best.pt
    run_script("slave_dali_ft", "slave.py", ["dali-ft"])

    # Stage 6: HPAC entropy model (~7 hr T4) -> hpac.pt
    run_script("hpac_train", "hpac.py",
               ["train", "--save", str(hpac_pt)])

    # Stage 7: PPMd-compress hpac.pt -> hpac.pt.ppmd
    stage_hpac_ppmd(hpac_pt, hpac_ppmd)

    # Stage 8: Pack everything into archive.zip
    run_script("archive_build", "archive.py",
               ["--master-pt", str(WORK / "v62_full_dali_swa.pt"),
                "--slave-pt",  str(WORK / "slave_v62_shrink_dali_best.pt"),
                "--hpac-ppmd", str(hpac_ppmd),
                "--sub-dir",   str(HERE)])

    if args.out_archive.exists():
        sz = args.out_archive.stat().st_size
        print(f"\n=== DONE === archive.zip = {sz:_} bytes ({sz / 1024:.1f} KB) "
              f"-> {args.out_archive}", flush=True)
    else:
        print(f"\nWARNING: {args.out_archive} not found after archive_build stage.",
              flush=True)


if __name__ == "__main__":
    main()
