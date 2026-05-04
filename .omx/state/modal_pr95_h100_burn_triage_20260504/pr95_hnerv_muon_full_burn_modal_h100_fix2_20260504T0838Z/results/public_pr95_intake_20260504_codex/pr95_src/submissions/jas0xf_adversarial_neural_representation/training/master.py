#!/usr/bin/env python
"""v62 master codec component — token-conditioned renderer with SCN QAT.

Defines the TokenRendererV62 architecture (3-layer 3x3 CNN over one-hot class
tokens with per-frame FiLM modulation and SCN bit-width QAT) and exposes two
training subcommands: `pretrain` runs the full-send PyAV-fed schedule, and
`dali-ft` runs the DALI-decoded GT continuation FT to close the train/eval gap.
"""
import sys, time, math, argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SAVE_DIR = Path(__file__).resolve().parent.parent / "training_workspace"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from data import (
    preload_rgb_pairs_av, SEGNET_IN_W, SEGNET_IN_H,
    CAMERA_W, CAMERA_H,
)
from frame_utils import DaliVideoDataset, segnet_model_input_size, camera_size
from modules import SegNet, segnet_sd_path

DEV = "cuda"
N = 600

# Architecture / SCN hyperparameters
B_INIT = 4.0
E_INIT = -3.0
B_MIN = 0.5
B_MAX = 8.0
NUM_CLASSES = 5
D_FILM = 8


def _scn_quantize(w, b, e, weight_dim_for_bcast):
    b_clip = b.clamp(B_MIN, B_MAX)
    shape = [1] * w.ndim
    shape[0] = -1
    b_view = b_clip.view(shape)
    e_view = e.view(shape)
    scale = torch.pow(2.0, e_view)
    max_q = torch.pow(2.0, b_view - 1) - 1
    min_q = -torch.pow(2.0, b_view - 1)
    q = torch.clamp(w / scale, min_q, max_q)
    q_round = q + (q.round() - q).detach()
    return q_round * scale


class SCNConv2d(nn.Module):
    def __init__(self, c_in, c_out, k, padding=0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(c_out, c_in, k, k))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        self.bias = nn.Parameter(torch.zeros(c_out)) if bias else None
        self.b = nn.Parameter(torch.full((c_out,), B_INIT))
        self.e = nn.Parameter(torch.full((c_out,), E_INIT))
        self.padding = padding
        self._scn_on = False
        self._w_per_ch = c_in * k * k

    def forward(self, x):
        w = _scn_quantize(self.weight, self.b, self.e, 0) if self._scn_on else self.weight
        return F.conv2d(x, w, self.bias, padding=self.padding)

    def total_bits(self):
        return (self.b.clamp(B_MIN, B_MAX) * self._w_per_ch).sum()


class SCNLinear(nn.Module):
    def __init__(self, in_f, out_f, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None
        self.b = nn.Parameter(torch.full((out_f,), B_INIT))
        self.e = nn.Parameter(torch.full((out_f,), E_INIT))
        self._scn_on = False
        self._w_per_ch = in_f

    def forward(self, x):
        w = _scn_quantize(self.weight, self.b, self.e, 0) if self._scn_on else self.weight
        return F.linear(x, w, self.bias)

    def total_bits(self):
        return (self.b.clamp(B_MIN, B_MAX) * self._w_per_ch).sum()


class TokenRendererV62(nn.Module):
    """Token-conditioned 3-layer CNN renderer with per-frame FiLM modulation.

    Pipeline:
      gt_tokens [N, 384, 512]  --one_hot-->  [N, 5, 384, 512]
      --> SCNConv2d(5,32,k3) --> GN --> FiLM(idx) --> GELU
      --> SCNConv2d(32,32,k3) --> GN --> GELU
      --> SCNConv2d(32,3,k3) --> sigmoid*255
      --> bilinear upsample to (CAMERA_H, CAMERA_W)
    """

    def __init__(self, num_pairs=N, num_classes=NUM_CLASSES, d_film=D_FILM):
        super().__init__()
        self.num_classes = num_classes
        self.frame_embed = nn.Embedding(num_pairs, d_film)
        nn.init.normal_(self.frame_embed.weight, std=0.02)
        self.film_gen = SCNLinear(d_film, 64)  # 64 = 32 scale + 32 shift
        nn.init.zeros_(self.film_gen.weight)   # FiLM starts as identity
        nn.init.zeros_(self.film_gen.bias)
        self.conv1 = SCNConv2d(num_classes, 32, k=3, padding=1)
        self.gn1 = nn.GroupNorm(8, 32)
        self.conv2 = SCNConv2d(32, 32, k=3, padding=1)
        self.gn2 = nn.GroupNorm(8, 32)
        self.out_conv = SCNConv2d(32, 3, k=3, padding=1)
        self.act = nn.GELU()

    def set_scn(self, on):
        for m in self.modules():
            if isinstance(m, (SCNConv2d, SCNLinear)):
                m._scn_on = on

    def forward(self, tokens, idx):
        # tokens: [B, H, W] long;  idx: [B] long
        x = F.one_hot(tokens, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        x = self.conv1(x)
        x = self.gn1(x)
        emb = self.frame_embed(idx)              # [B, d_film]
        film = self.film_gen(emb)                # [B, 64]
        scale, shift = film.chunk(2, dim=1)      # [B, 32], [B, 32]
        x = x * (1.0 + scale.view(-1, 32, 1, 1)) + shift.view(-1, 32, 1, 1)
        x = self.act(x)
        x = self.act(self.gn2(self.conv2(x)))
        x = self.out_conv(x)
        raw = torch.sigmoid(x) * 255.0
        return F.interpolate(raw, size=(CAMERA_H, CAMERA_W),
                             mode="bilinear", align_corners=False)

    def scn_total_bits(self):
        total = self.conv1.total_bits() + self.conv2.total_bits() + self.out_conv.total_bits() + self.film_gen.total_bits()
        return total

    def scn_total_weight_count(self):
        n = 0
        for m in self.modules():
            if isinstance(m, (SCNConv2d, SCNLinear)):
                n += m.weight.numel()
        return n

    def avg_bits(self):
        per_layer = []
        for name, m in self.named_modules():
            if isinstance(m, (SCNConv2d, SCNLinear)):
                per_layer.append((name, float(m.b.clamp(B_MIN, B_MAX).mean())))
        return per_layer


# Pretrain (full-send) hyperparameters
PRETRAIN_BS = 32
PRETRAIN_EPOCHS_CAP = 200
PRETRAIN_PATIENCE = 20
PRETRAIN_LR = 1e-4
PRETRAIN_LR_FRAME_EMB = 1e-3
PRETRAIN_TEMP_START = 1.0
PRETRAIN_TEMP_END = 0.1
PRETRAIN_ERR_BOOST_LOW = 1.0
PRETRAIN_ERR_BOOST_HIGH = 1.0
PRETRAIN_BOOST_HIGH_FROM = 100
PRETRAIN_SCN_FROM = 50
PRETRAIN_LAMBDA_BITS_FINAL = 1e-6
PRETRAIN_SWA_FROM = 100
PRETRAIN_EVAL_EVERY = 10
CKPT_DIR = SAVE_DIR / "v62_full_ckpts"


def cmd_pretrain():
    """Full-send pretrain with PyAV-decoded GT, SCN ramp, and tail SWA.

    Three-phase schedule (FP warmup → SCN-on weight compression → tail SWA).
    Specific epoch boundaries and ramp coefficients are tuned and intentionally
    unspecified while the competition is active.
    """
    BS = PRETRAIN_BS
    EPOCHS_CAP = PRETRAIN_EPOCHS_CAP
    PATIENCE = PRETRAIN_PATIENCE
    LR = PRETRAIN_LR
    LR_FRAME_EMB = PRETRAIN_LR_FRAME_EMB
    TEMP_START = PRETRAIN_TEMP_START
    TEMP_END = PRETRAIN_TEMP_END
    ERR_BOOST_LOW = PRETRAIN_ERR_BOOST_LOW
    ERR_BOOST_HIGH = PRETRAIN_ERR_BOOST_HIGH
    BOOST_HIGH_FROM = PRETRAIN_BOOST_HIGH_FROM
    SCN_FROM = PRETRAIN_SCN_FROM
    LAMBDA_BITS_FINAL = PRETRAIN_LAMBDA_BITS_FINAL
    SWA_FROM = PRETRAIN_SWA_FROM
    EVAL_EVERY = PRETRAIN_EVAL_EVERY

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[v62full] BS={BS}  cap={EPOCHS_CAP}  patience={PATIENCE}  SCN@{SCN_FROM}  SWA@{SWA_FROM}", flush=True)

    gen = TokenRendererV62().to(DEV)
    print(f"[v62full] Total params: {sum(p.numel() for p in gen.parameters()):_}", flush=True)

    segnet = SegNet().eval().to(DEV)
    segnet.load_state_dict(load_file(segnet_sd_path, device=DEV))
    for p in segnet.parameters():
        p.requires_grad = False

    print("[v62full] preload + gt_tokens...", flush=True)
    rgb_pairs = preload_rgb_pairs_av(ROOT / "videos", ROOT / "public_test_video_names.txt")
    gt_tokens = torch.empty(N, SEGNET_IN_H, SEGNET_IN_W, dtype=torch.long, device=DEV)
    with torch.inference_mode():
        for s in range(0, N, BS):
            e = min(s + BS, N)
            x = rgb_pairs[s:e].to(DEV).float().permute(0, 1, 4, 2, 3)
            gt_tokens[s:e] = segnet(segnet.preprocess_input(x[:, 1:2])).argmax(1)
    del rgb_pairs

    @torch.no_grad()
    def eval_seg(g):
        correct = 0; total = 0
        for s in range(0, N, BS):
            e = min(s + BS, N)
            idx = torch.arange(s, e, device=DEV)
            tok = gt_tokens[s:e]
            m = g(tok, idx)
            x = m.round().clamp(0, 255).unsqueeze(1)
            preds = segnet(segnet.preprocess_input(x)).argmax(1)
            correct += (preds == tok).float().sum().item()
            total += preds.numel()
        return 1.0 - correct / total

    init_seg = eval_seg(gen)
    print(f"[v62full] init seg: {init_seg:.6f}", flush=True)

    frame_emb_params = [gen.frame_embed.weight]
    other_params = [p for n, p in gen.named_parameters() if n != "frame_embed.weight"]
    opt = torch.optim.AdamW(
        [
            {"params": other_params, "lr": LR},
            {"params": frame_emb_params, "lr": LR_FRAME_EMB},
        ],
        eps=1e-8,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_CAP, eta_min=LR * 0.02)

    best_seg = init_seg
    best_state = {k: v.detach().clone().cpu() for k, v in gen.state_dict().items()}
    best_ep = -1
    swa_buf = None; swa_count = 0
    last_improve_ep = 0
    t0 = time.time()

    for ep in range(EPOCHS_CAP):
        gen.train()
        if ep == SCN_FROM:
            print(f"[v62full] enabling SCN at ep {ep}", flush=True)
            gen.set_scn(True)
        progress = ep / max(EPOCHS_CAP - 1, 1)
        temp = TEMP_START * (TEMP_END / TEMP_START) ** progress
        if ep < BOOST_HIGH_FROM:
            err_boost = ERR_BOOST_LOW
        else:
            t = (ep - BOOST_HIGH_FROM) / max(EPOCHS_CAP - BOOST_HIGH_FROM, 1)
            err_boost = ERR_BOOST_LOW + t * (ERR_BOOST_HIGH - ERR_BOOST_LOW)
        lam_bits = (
            0.0 if ep < SCN_FROM
            else LAMBDA_BITS_FINAL * (ep - SCN_FROM) / max(EPOCHS_CAP - SCN_FROM, 1)
        )

        perm = torch.randperm(N, device=DEV)
        for s in range(0, N, BS):
            e = min(s + BS, N)
            idx = perm[s:e]
            tok = gt_tokens[idx]
            m = gen(tok, idx)
            m_q = m + (m.round() - m).detach()
            x = m_q.unsqueeze(1)
            logits = segnet(segnet.preprocess_input(x))
            ce = F.cross_entropy(logits / temp, tok, reduction='none')
            with torch.no_grad():
                wrong = (logits.argmax(1) != tok).float()
                boost = 1.0 + wrong * err_boost
            seg_loss = (ce * boost).mean()
            loss = seg_loss
            if lam_bits > 0:
                bits_loss = lam_bits * gen.scn_total_bits() / 12032
                loss = loss + bits_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
            opt.step()
        sched.step()

        if ep >= SWA_FROM:
            if swa_buf is None:
                swa_buf = {k: v.detach().clone().float() for k, v in gen.state_dict().items() if v.dtype.is_floating_point}
                swa_count = 1
            else:
                for k, v in gen.state_dict().items():
                    if v.dtype.is_floating_point and k in swa_buf:
                        swa_buf[k].mul_(swa_count / (swa_count + 1)).add_(v.detach().float() / (swa_count + 1))
                swa_count += 1

        # Per-epoch checkpoint (small)
        torch.save({k: v.detach().cpu() for k, v in gen.state_dict().items()},
                   CKPT_DIR / f"ep_{ep+1:04d}.pt")

        if (ep + 1) % EVAL_EVERY == 0 or ep == 0 or ep == SCN_FROM:
            gen.eval()
            seg = eval_seg(gen)
            with torch.no_grad():
                avg_b = {n.split('.')[-1]: float(m.b.clamp(B_MIN, B_MAX).mean())
                         for n, m in gen.named_modules()
                         if isinstance(m, (SCNConv2d, SCNLinear))}
                tb = float(gen.scn_total_bits())
            improved = seg < best_seg
            if improved:
                best_seg = seg
                best_state = {k: v.detach().clone().cpu() for k, v in gen.state_dict().items()}
                best_ep = ep
                last_improve_ep = ep
                torch.save({"state_dict": best_state, "best_seg": best_seg,
                            "ep": ep, "kind": "v62_full_best"},
                           SAVE_DIR / "v62_full_best.pt")
            avgs = " ".join(f"{n}={v:.2f}" for n, v in avg_b.items())
            stale = ep - last_improve_ep
            print(f"[v62full] ep {ep+1:>4d}/{EPOCHS_CAP}  temp={temp:.3f}  boost=x{err_boost:.0f}  "
                  f"lam={lam_bits:.1e}  seg={seg:.6f}  best={best_seg:.6f}@ep{best_ep+1}  "
                  f"stale={stale}  bits[{avgs}]  scnB={tb/8:.0f}  "
                  f"dt={time.time()-t0:.0f}s",
                  flush=True)

    gen.eval()
    final_seg = eval_seg(gen)
    print(f"\n[v62full] FINAL: seg={final_seg:.6f}  best={best_seg:.6f}@ep{best_ep+1}  init={init_seg:.6f}",
          flush=True)

    torch.save({"state_dict": {k: v.detach().cpu() for k, v in gen.state_dict().items()},
                "final_seg": final_seg, "best_seg": best_seg, "kind": "v62_full_last"},
               SAVE_DIR / "v62_full_last.pt")

    if swa_buf is not None and swa_count > 1:
        backup = {k: v.detach().clone() for k, v in gen.state_dict().items()}
        swa_state = dict(backup)
        for k, v in swa_buf.items():
            swa_state[k] = v.to(backup[k].device, backup[k].dtype)
        gen.load_state_dict(swa_state)
        gen.eval()
        swa_seg = eval_seg(gen)
        print(f"[v62full] SWA-{swa_count} seg: {swa_seg:.6f}", flush=True)
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in swa_state.items()},
                    "swa_seg": swa_seg, "kind": "v62_full_swa"},
                   SAVE_DIR / "v62_full_swa.pt")

    print(f"[v62full] DONE — total wall {time.time()-t0:.0f}s", flush=True)


INIT_CKPT = SAVE_DIR / "v62_full_best.pt"


def gt_dali_load(bs=8):
    """Return (N, 2, H, W, 3) uint8 of DALI-decoded GT pairs on CPU."""
    print("[md-ft] loading GT via DALI/NVDEC...", flush=True)
    with open(ROOT / "public_test_video_names.txt") as f:
        names = [line.strip() for line in f.readlines()]
    ds = DaliVideoDataset(names, data_dir=ROOT / "videos",
                          batch_size=bs, device=torch.device("cuda"))
    ds.prepare_data()
    parts = []
    for _, _, batch in ds:
        parts.append(batch.detach().cpu())
    all_frames = torch.cat(parts, dim=0)
    assert all_frames.shape[0] >= N, f"DALI got {all_frames.shape[0]}, need {N}"
    return all_frames[:N].contiguous()


def cmd_dali_ft(argv):
    """DALI-decoded GT continuation FT.

    Loads GT pairs once via DaliVideoDataset (matches the official evaluator),
    computes gt_tokens = segnet(DALI-GT) used as both input AND reference, and
    keeps SCN ON throughout so weights stay bit-exact to the ship export
    (FP16 holds q*2^e exactly -> ship_master == train_master at every pixel).
    """
    ap = argparse.ArgumentParser(prog="master.py dali-ft")
    ap.add_argument("--init", type=str, default=str(INIT_CKPT))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-emb", type=float, default=5e-4)
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--boost", type=float, default=4.0)
    ap.add_argument("--lam-bits", type=float, default=1e-6,
                    help="SCN bits regularizer; small to allow tiny size drift while keeping grid stable.")
    ap.add_argument("--swa-from", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--tag", type=str, default="dali")
    args = ap.parse_args(argv)

    save_best = SAVE_DIR / f"v62_full_{args.tag}_best.pt"
    save_last = SAVE_DIR / f"v62_full_{args.tag}_last.pt"
    save_swa  = SAVE_DIR / f"v62_full_{args.tag}_swa.pt"

    print(f"[md-ft] init={Path(args.init).name} epochs={args.epochs} bs={args.bs} "
          f"lr={args.lr} lr_emb={args.lr_emb} temp={args.temp} boost={args.boost} "
          f"lam_bits={args.lam_bits} swa_from={args.swa_from}", flush=True)

    segnet = SegNet().eval().to(DEV)
    segnet.load_state_dict(load_file(segnet_sd_path, device=DEV))
    for p in segnet.parameters():
        p.requires_grad = False

    # GT pairs via DALI (matches official evaluator pixel pipeline)
    gt_pairs = gt_dali_load(bs=args.bs)
    H_in, W_in = segnet_model_input_size[1], segnet_model_input_size[0]
    print(f"[md-ft] computing gt_tokens at {H_in}x{W_in} from DALI GT...", flush=True)
    gt_tokens = torch.empty(N, H_in, W_in, dtype=torch.long, device=DEV)
    with torch.inference_mode():
        for s in range(0, N, args.bs):
            e = min(s + args.bs, N)
            x = gt_pairs[s:e].to(DEV).float().permute(0, 1, 4, 2, 3)  # (B,2,3,H,W)
            gt_tokens[s:e] = segnet(segnet.preprocess_input(x)).argmax(1)
    del gt_pairs
    torch.cuda.empty_cache()

    # Master with SCN ON (matches ship export bit-exactly via q*2^e in FP16)
    gen = TokenRendererV62(num_pairs=N).to(DEV)
    blob = torch.load(args.init, map_location=DEV, weights_only=False)
    sd = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    gen.load_state_dict(sd, strict=True)
    gen.set_scn(True)
    n_w = gen.scn_total_weight_count()
    n_total = sum(p.numel() for p in gen.parameters())
    print(f"[md-ft] params: total={n_total:_} SCN_w={n_w:_} init_meta_seg={blob.get('best_seg', '?')}",
          flush=True)

    @torch.no_grad()
    def eval_seg(g):
        g.eval()
        correct = 0; total = 0
        for s in range(0, N, args.bs):
            e = min(s + args.bs, N)
            idx = torch.arange(s, e, device=DEV)
            tok = gt_tokens[s:e]
            m = g(tok, idx)
            x = m.round().clamp(0, 255).unsqueeze(1)
            preds = segnet(segnet.preprocess_input(x)).argmax(1)
            correct += (preds == tok).float().sum().item()
            total += preds.numel()
        return 1.0 - correct / total

    init_seg = eval_seg(gen)
    print(f"[md-ft] init seg (DALI refs, SCN on): {init_seg:.6f}", flush=True)

    frame_emb_params = [gen.frame_embed.weight]
    other_params = [p for n, p in gen.named_parameters() if n != "frame_embed.weight"]
    opt = torch.optim.AdamW(
        [{"params": other_params, "lr": args.lr},
         {"params": frame_emb_params, "lr": args.lr_emb}],
        eps=1e-8,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)

    best_seg = init_seg
    best_ep = -1
    swa_buf = None; swa_count = 0
    t0 = time.time()
    for ep in range(args.epochs):
        gen.train()
        perm = torch.randperm(N, device=DEV)
        ep_ce = 0.0; ep_cnt = 0
        for s in range(0, N, args.bs):
            e = min(s + args.bs, N)
            idx = perm[s:e]
            tok = gt_tokens[idx]
            m = gen(tok, idx)
            m_q = m + (m.round() - m).detach()
            x = m_q.unsqueeze(1)
            logits = segnet(segnet.preprocess_input(x))
            ce = F.cross_entropy(logits / args.temp, tok, reduction='none')
            with torch.no_grad():
                wrong = (logits.argmax(1) != tok).float()
                boost = 1.0 + wrong * args.boost
            seg_loss = (ce * boost).mean()
            loss = seg_loss
            if args.lam_bits > 0:
                loss = loss + args.lam_bits * gen.scn_total_bits() / max(n_w, 1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
            opt.step()
            ep_ce += float(seg_loss) * tok.shape[0]
            ep_cnt += tok.shape[0]
        sched.step()

        if ep >= args.swa_from:
            if swa_buf is None:
                swa_buf = {k: v.detach().clone().float() for k, v in gen.state_dict().items()
                           if v.dtype.is_floating_point}
                swa_count = 1
            else:
                for k, v in gen.state_dict().items():
                    if v.dtype.is_floating_point and k in swa_buf:
                        swa_buf[k].mul_(swa_count / (swa_count + 1)).add_(v.detach().float() / (swa_count + 1))
                swa_count += 1

        if (ep + 1) % args.eval_every == 0 or ep == 0 or ep == args.epochs - 1:
            seg = eval_seg(gen)
            improved = seg < best_seg
            if improved:
                best_seg = seg; best_ep = ep
                torch.save({"state_dict": {k: v.detach().cpu() for k, v in gen.state_dict().items()},
                            "best_seg": best_seg, "ep": ep, "kind": f"v62_full_{args.tag}_best",
                            "gt_decoder": "dali"},
                           save_best)
            scn_kb = float(gen.scn_total_bits()) / 8 / 1024
            tag_s = " *NEW BEST*" if improved else ""
            print(f"[md-ft] ep {ep+1:>4d}/{args.epochs}  seg={seg:.6f}  "
                  f"best={best_seg:.6f}@ep{best_ep+1}  init={init_seg:.6f}  "
                  f"scn_KB={scn_kb:.2f}  dt={time.time()-t0:.0f}s{tag_s}",
                  flush=True)

    gen.eval()
    final_seg = eval_seg(gen)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in gen.state_dict().items()},
                "final_seg": final_seg, "best_seg": best_seg,
                "kind": f"v62_full_{args.tag}_last", "gt_decoder": "dali"},
               save_last)

    if swa_buf is not None and swa_count > 1:
        backup = {k: v.detach().clone() for k, v in gen.state_dict().items()}
        swa_state = dict(backup)
        for k, v in swa_buf.items():
            swa_state[k] = v.to(backup[k].device, backup[k].dtype)
        gen.load_state_dict(swa_state)
        gen.eval()
        swa_seg = eval_seg(gen)
        print(f"[md-ft] SWA-{swa_count} seg: {swa_seg:.6f}", flush=True)
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in swa_state.items()},
                    "swa_seg": swa_seg, "kind": f"v62_full_{args.tag}_swa",
                    "gt_decoder": "dali"},
                   save_swa)

    print(f"\n[md-ft] DONE init={init_seg:.6f} best={best_seg:.6f}@ep{best_ep+1} "
          f"final={final_seg:.6f} wall={time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(prog="master.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pretrain", help="full-send PyAV-fed pretrain (SCN ramp + tail SWA)")
    sub.add_parser("dali-ft", help="DALI-decoded GT continuation FT (forwards remaining args)")

    # Parse only the subcommand and forward the rest to dali-ft (which has its own argparse).
    if len(sys.argv) < 2:
        ap.print_help()
        sys.exit(2)
    cmd = sys.argv[1]
    rest = sys.argv[2:]
    if cmd == "pretrain":
        # Reject extra args for pretrain (no argparse inside).
        ap.parse_args([cmd])
        cmd_pretrain()
    elif cmd == "dali-ft":
        cmd_dali_ft(rest)
    else:
        ap.parse_args([cmd])  # triggers argparse error


if __name__ == "__main__":
    main()
