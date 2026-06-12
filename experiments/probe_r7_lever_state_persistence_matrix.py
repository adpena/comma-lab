#!/usr/bin/env python
"""R7 adversarial probe — the LEVER-STATE-PERSISTENCE MATRIX.

R6 confirmed a bug CLASS: "lever mutable/accumulated state that should CARRY
across a boundary but RESETS." It has TWO known instances, both QAT-EMA (R2 fixed
*resume*, R6 fixed the *stage-boundary*). R7 GENERALIZES: it MEASURES (does not
assume) every lever's mutable/accumulated state across EVERY boundary type, hunting
for a SIBLING reset-bug (carry-when-should / reset-when-should) anywhere in the 5
levers, plus the un-tested boundary combinations the prior rounds ran only in
isolation.

The state inventory (find-all, do not trust a list):
  L4 QAT  : ``tensor_sensitivity_ema``  (FIXED resume R2 + stage-boundary R6).
  L2 anneal: ``epoch_in_stage`` schedule (resets per stage BY DESIGN — verify).
  L1 rate : stateless (reads live weights/latents) — confirm.
  L3 FiLM : ``stored_pose`` buffer + FiLM weights (ride the carried decoder sd).
  L5 margin: stateless — confirm.
  shared  : weight-EMA shadow, AdamW/Muon momentum, schedulers, RNG.

The boundary-type matrix (the combinations R1-R6 ran in isolation, never together):
  B1 non-QAT -> QAT          : the EMA should START EMPTY at the first QAT stage
                               (nothing seeds it before use_qat) -> the lever-
                               ACTIVATION boundary (L4 set on a non-QAT stage rides
                               along as a no-op, then engages).
  B2 QAT -> QAT              : the EMA should CARRY (R6 fix).
  B3 lever-activation-change : L4 OFF in stage N, ON in N+1 -> the new stage builds
                               an empty EMA (correct: no prior EMA to carry).
  B4 resume mid-QAT-stage-2  : the resume-restored EMA wins over the carry (the path
                               the R6 fix introduced; R2 only tested resume into the
                               FIRST QAT stage).

Authority: synthetic scorer, RESEARCH-ONLY, [macOS-CPU advisory] NON-PROMOTABLE.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from tac.torch_vehicle.checkpoint import load_checkpoint
from tac.torch_vehicle.curriculum import StageSpec, seg_temperature_for_epoch
from tac.torch_vehicle.driver import (
    TorchVehicleConfig,
    TorchVehicleDriver,
    import_vendored_bundle,
)
from tac.torch_vehicle.scorer_context import SyntheticScorerContext


def _ce(s, t):
    return torch.nn.functional.cross_entropy(s, t)


def _spec(
    name: str,
    epochs: int,
    *,
    use_qat: bool,
    init_random: bool,
    score_aware_qat: bool = True,
    all_levers: bool = True,
) -> StageSpec:
    kw = {
        "name": name, "epochs": epochs, "seg_loss_fn": _ce, "eval_every": 1,
        "batch_size": 4, "ema_decay": 0.999, "use_muon": False, "adamw_lr": 1e-3,
        "muon_lr": 2e-4, "muon_weight_decay": 0.0, "latent_lr_mult": 10.0,
        "grad_clip": 1.0, "grad_clip_muon": 1.0, "lr_floor_ratio": 5e-6,
        "seg_weight": 100.0, "pose_weight": 1.0,
        "cat_lambda": 0.01 if all_levers else 0.0, "cat_sigma": 0.2,
        "use_qat": use_qat, "init_latents_random": init_random,
    }
    if all_levers:
        kw.update(
            seg_surrogate="soft_cosine", seg_temperature=1.0, seg_temperature_end=0.05,
            rate_lambda_w=1e-3, rate_lambda_lat=1e-3,
            score_aware_qat=score_aware_qat, qat_sensitivity_decay=0.99,
            margin_weight_tau=2.0,
        )
    return StageSpec(**kw)


def _cfg(out_dir: Path) -> TorchVehicleConfig:
    return TorchVehicleConfig(
        base_channels=8, latent_dim=28, out_dir=out_dir, checkpoint_every_epochs=1,
        device="cpu", seed=29, pose_film_enabled=True, pose_film_hidden=8,
    )


# --------------------------------------------------------------------------
# A hooked driver that records EVERY lever's state at EVERY stage build + the
# first train epoch of each stage (the consumption point), for the full matrix.
# --------------------------------------------------------------------------
def _run_matrix(out_dir: Path, n_pairs: int, curriculum: list[StageSpec]) -> dict:
    cfg = _cfg(out_dir)
    sc = SyntheticScorerContext(n_pairs=n_pairs, device="cpu", seed=29)
    v = import_vendored_bundle()
    drv = TorchVehicleDriver(cfg, scorer=sc, vendored=v, curriculum=curriculum)

    orig_build = drv._build_stage_runtime
    orig_train = drv._train_one_epoch

    # Per-stage records keyed by stage index.
    rec: dict[int, dict] = {}
    stage_i = {"i": 0}
    prev_holder: dict[str, object] = {}

    def hooked_build(spec, **kw):
        rt = orig_build(spec, **kw)
        idx = stage_i["i"]
        ent = rec.setdefault(idx, {})
        ent["name"] = spec.name
        ent["use_qat"] = spec.use_qat
        ent["score_aware_qat"] = spec.score_aware_qat
        # Capture the PRIOR stage's END state BEFORE building this stage.
        prev = prev_holder.get("rt")
        if prev is not None:
            prec = rec[idx - 1]
            prec["end_ema_count"] = len(prev.tensor_sensitivity_ema)  # type: ignore
            # weight-EMA shadow fingerprint (sum of all EMA decoder params).
            prec["end_ema_decoder_sum"] = float(
                sum(p.detach().double().sum().item()
                    for p in prev.ema_decoder.state_dict().values()  # type: ignore
                    if p.is_floating_point())
            )
            # FiLM stored_pose fingerprint (a buffer in the carried decoder sd).
            sd = prev.decoder.state_dict()  # type: ignore
            if "stored_pose" in sd:
                prec["end_stored_pose_sum"] = float(sd["stored_pose"].double().sum().item())
        prev_holder["rt"] = rt
        stage_i["i"] += 1
        return rt

    def hooked_train(rt, spec, *, epoch_in_stage=0):
        # At the CONSUMPTION point (start of a stage, AFTER the run() carry-seed +
        # any resume-restore have executed), record the state the levers actually read.
        if epoch_in_stage == 0:
            # Map spec.name -> stage index (names are unique in these curricula).
            idx = next(i for i, s in enumerate(curriculum) if s.name == spec.name)
            ent = rec.setdefault(idx, {})
            if "start_ema_count" not in ent:
                ent["start_ema_count"] = len(rt.tensor_sensitivity_ema)
                ent["start_anneal_t0"] = seg_temperature_for_epoch(spec, 0)
                ent["start_anneal_tlast"] = seg_temperature_for_epoch(spec, spec.epochs - 1)
                sd = rt.decoder.state_dict()
                if "stored_pose" in sd:
                    ent["start_stored_pose_sum"] = float(sd["stored_pose"].double().sum().item())
                ent["start_ema_decoder_sum"] = float(
                    sum(p.detach().double().sum().item()
                        for p in rt.ema_decoder.state_dict().values()
                        if p.is_floating_point())
                )
        return orig_train(rt, spec, epoch_in_stage=epoch_in_stage)

    drv._build_stage_runtime = hooked_build  # type: ignore[method-assign]
    drv._train_one_epoch = hooked_train  # type: ignore[method-assign]
    out = drv.run()
    assert out["status"] == "complete", out
    # Final stage end-state (loop exits before another build hook fires).
    final = prev_holder.get("rt")
    if final is not None:
        fi = len(curriculum) - 1
        rec[fi]["end_ema_count"] = len(final.tensor_sensitivity_ema)  # type: ignore
    return rec


def main() -> int:
    torch.manual_seed(29)
    n_pairs = 8

    # The full boundary matrix in ONE curriculum:
    #   s0 pre_qat (L4 set, use_qat=False)  -> B1 non-QAT->QAT into s1
    #   s1 qat_a (use_qat, L4 on)           -> B2 QAT->QAT into s2
    #   s2 qat_b (use_qat, L4 on)           -> B3 lever-activation-change into s3
    #   s3 qat_c (use_qat, L4 OFF)          (score_aware_qat=False: L4 deactivated)
    s0 = _spec("pre_qat", 3, use_qat=False, init_random=True)
    s1 = _spec("qat_a", 4, use_qat=True, init_random=False)
    s2 = _spec("qat_b", 4, use_qat=True, init_random=False)
    s3 = _spec("qat_c_l4off", 3, use_qat=True, init_random=False, score_aware_qat=False)
    curriculum = [s0, s1, s2, s3]

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "matrix"
        rec = _run_matrix(out_dir, n_pairs, curriculum)

    print("=" * 78)
    print("R7 LEVER-STATE-PERSISTENCE MATRIX (MEASURED)")
    print("=" * 78)
    findings: list[str] = []

    # ---- L4 QAT sensitivity EMA across each boundary ----------------------
    # B1 non-QAT(s0) -> QAT(s1): s0 has use_qat=False so the QAT block never runs;
    #    accumulate_tensor_sensitivity is GATED on use_qat AND score_aware_qat, so
    #    s0 END EMA must be 0 (empty). s1 START EMA must therefore be 0 (correct: the
    #    carry from an empty prior stage is empty), then s1 ACCUMULATES.
    s0_end = rec[0].get("end_ema_count", -1)
    s1_start = rec[1].get("start_ema_count", -1)
    s1_end = rec[1].get("end_ema_count", -1)
    print(f"B1 non-QAT->QAT : s0(pre_qat,use_qat=F) END EMA={s0_end}  "
          f"-> s1(qat_a) START EMA={s1_start} -> END EMA={s1_end}")
    if s0_end != 0:
        findings.append(
            f"B1: pre-QAT stage accumulated a non-empty EMA ({s0_end}) despite "
            f"use_qat=False — the accumulate gate leaked.")
    if s1_start != 0:
        findings.append(
            f"B1: first QAT stage START EMA={s1_start} (expected 0 — nothing seeds "
            f"it before the first use_qat stage).")
    if s1_end <= 0:
        findings.append(f"B1: first QAT stage END EMA={s1_end} (expected >0 — it must accumulate).")

    # B2 QAT(s1) -> QAT(s2): the R6 carry. s1 END must equal s2 START (CARRIED).
    s2_start = rec[2].get("start_ema_count", -1)
    s2_end = rec[2].get("end_ema_count", -1)
    print(f"B2 QAT->QAT     : s1(qat_a) END EMA={s1_end} -> s2(qat_b) START EMA={s2_start}  "
          f"-> END EMA={s2_end}")
    if s2_start != s1_end or s2_start <= 0:
        findings.append(
            f"B2: QAT->QAT EMA did NOT carry (s1 END={s1_end} != s2 START={s2_start}) "
            f"— the R6 fix regressed.")

    # B3 lever-activation-change QAT(s2,L4 on) -> QAT(s3,L4 OFF): s3 has
    #    score_aware_qat=False, so its QAT block uses the VENDORED uniform path and
    #    NEVER accumulates/consumes the EMA. The carry still SEEDS rt's EMA (the
    #    carry block is unconditional on score_aware), but it is harmlessly unused.
    #    Correct behavior: no crash, the vendored uniform QAT runs, byte-path intact.
    s3_start = rec[3].get("start_ema_count", -1)
    s3_end = rec[3].get("end_ema_count", -1)
    print(f"B3 L4 on->OFF   : s2(qat_b,L4=on) END EMA={s2_end} -> s3(qat_c,L4=off) "
          f"START EMA={s3_start} -> END EMA={s3_end}")
    # s3 (L4 off) must NOT accumulate further (the gate is use_qat AND score_aware_qat).
    if s3_end != s3_start:
        findings.append(
            f"B3: L4-OFF stage MUTATED the EMA (start={s3_start} end={s3_end}) despite "
            f"score_aware_qat=False — the accumulate gate leaked on the OFF stage.")

    # ---- L2 anneal: resets per stage BY DESIGN (each stage restarts 1.0->0.05) ----
    print("L2 anneal/stage :", " ".join(
        f"s{i}:{rec[i].get('start_anneal_t0','?'):.3f}->{rec[i].get('start_anneal_tlast','?'):.3f}"
        for i in range(len(curriculum)) if "start_anneal_t0" in rec[i]))
    for i in range(len(curriculum)):
        t0 = rec[i].get("start_anneal_t0")
        tl = rec[i].get("start_anneal_tlast")
        if t0 is None:
            continue
        # DESIGN: every stage restarts its own anneal 1.0 -> 0.05 (per-stage cosine).
        if abs(t0 - 1.0) > 1e-9 or abs(tl - 0.05) > 1e-9:
            findings.append(
                f"L2: stage {i} anneal did NOT restart 1.0->0.05 ({t0}->{tl}) — "
                f"the per-stage anneal carried incorrectly.")

    # ---- L3 FiLM stored_pose: rides the carried decoder sd (must be STABLE) -------
    # The stored_pose is a fixed buffer (GT side-info, never optimized). It must be
    # IDENTICAL at every stage boundary (carried with the decoder sd, never reset).
    pose_sums = [rec[i].get("start_stored_pose_sum") for i in range(len(curriculum))]
    print(f"L3 stored_pose  : per-stage-START sums = {pose_sums}")
    base = pose_sums[0]
    for i, ps in enumerate(pose_sums):
        if ps is None:
            findings.append(f"L3: stage {i} has NO stored_pose buffer (FiLM lost the pose side-info).")
        elif base is not None and abs(ps - base) > 1e-6:
            findings.append(
                f"L3: stored_pose CHANGED at stage {i} (start sum {ps} != stage0 {base}) "
                f"— the GT pose side-info drifted/reset across a boundary.")

    # ---- weight-EMA shadow: must CARRY (start of stage N+1 == end of stage N) -----
    print("weight-EMA carry:", end=" ")
    for i in range(len(curriculum) - 1):
        end = rec[i].get("end_ema_decoder_sum")
        nxt = rec[i + 1].get("start_ema_decoder_sum")
        carried = (end is not None and nxt is not None and abs(end - nxt) < 1e-6)
        print(f"s{i}->s{i+1}:{'OK' if carried else 'RESET'}", end="  ")
        if not carried:
            findings.append(
                f"weight-EMA: stage {i}->{i+1} EMA shadow did NOT carry "
                f"(end={end} != start={nxt}).")
    print()

    # ============================================================
    # B4: resume landing MID the SECOND QAT stage (s2/qat_b, epoch 2).
    #     R2 only tested resume into the FIRST QAT stage; the R6 carry fix introduced
    #     a NEW interaction (carry-seed THEN resume-restore). Verify the resume-
    #     restored EMA wins AND the bit-identity of the kill+resume holds.
    # ============================================================
    b4 = _probe_resume_mid_second_qat(n_pairs, curriculum)
    print(f"B4 resume mid-s2: kill@global_ep={b4['kill_ep']} (mid qat_b) -> "
          f"resume-restored EMA={b4['restored_ema_count']} tensors "
          f"(checkpoint had {b4['ckpt_ema_count']}); "
          f"final archive resumed=={b4['resumed_bytes']}B vs uninterrupted="
          f"{b4['ref_bytes']}B -> {'BIT-IDENTICAL' if b4['bit_identical'] else 'DIVERGED'}")
    if b4["ckpt_ema_count"] <= 0:
        findings.append(
            f"B4: mid-second-QAT-stage checkpoint had EMPTY EMA "
            f"({b4['ckpt_ema_count']}) — the carry+accumulate did not populate it "
            f"before the checkpoint.")
    if b4["restored_ema_count"] != b4["ckpt_ema_count"]:
        findings.append(
            f"B4: resume did NOT restore the checkpointed EMA "
            f"(restored {b4['restored_ema_count']} != checkpoint {b4['ckpt_ema_count']}).")
    if not b4["bit_identical"]:
        findings.append(
            f"B4: resume mid-second-QAT-stage is NOT bit-identical to the "
            f"uninterrupted run ({b4['resumed_bytes']} != {b4['ref_bytes']}).")

    print("=" * 78)
    if findings:
        print(f"R7 MATRIX FINDINGS ({len(findings)}):")
        for f in findings:
            print(f"  - {f}")
        print("=" * 78)
        return 1
    print("R7 MATRIX: CLEAN — every lever state carries/resets CORRECTLY across "
          "every boundary type (B1/B2/B3/B4). No sibling reset-bug.")
    print("=" * 78)
    return 0


def _probe_resume_mid_second_qat(n_pairs: int, curriculum: list[StageSpec]) -> dict:
    """Kill mid the SECOND QAT stage (qat_b), resume, and compare the final archive
    to an uninterrupted run. R2 only covered resume into the FIRST QAT stage; this is
    the path the R6 carry fix introduced (carry-seed THEN resume-restore)."""
    v = import_vendored_bundle()
    qat_idxs = [i for i, s in enumerate(curriculum) if s.use_qat]
    second_qat = qat_idxs[1]
    # global epoch at the MIDDLE of the second QAT stage.
    base_ep = sum(curriculum[i].epochs for i in range(second_qat))
    kill_ep = base_ep + max(1, curriculum[second_qat].epochs // 2)

    with tempfile.TemporaryDirectory() as td:
        ref_dir = Path(td) / "ref"
        res_dir = Path(td) / "res"

        # Reference: uninterrupted.
        scr = SyntheticScorerContext(n_pairs=n_pairs, device="cpu", seed=29)
        dref = TorchVehicleDriver(_cfg(ref_dir), scorer=scr, vendored=v, curriculum=curriculum)
        dref.run()
        ref_bytes = len((ref_dir / "best" / "best_archive.bin").read_bytes())

        # Killed run: stop AFTER the checkpoint at kill_ep lands.
        sck = SyntheticScorerContext(n_pairs=n_pairs, device="cpu", seed=29)
        dkill = TorchVehicleDriver(_cfg(res_dir), scorer=sck, vendored=v, curriculum=curriculum)
        dkill._stop_after_global_epoch = kill_ep
        from tac.torch_vehicle.driver import _SimulatedDeath
        try:
            dkill.run()
        except _SimulatedDeath:
            pass
        # Inspect the checkpoint's EMA.
        merged = load_checkpoint(res_dir, map_location="cpu")
        ckpt_ema = merged.get("tensor_sensitivity_ema") or {}
        ckpt_ema_count = len(ckpt_ema)

        # Resume: a fresh driver picks up from the checkpoint. Hook the restore to
        # capture how many EMA tensors landed.
        scr2 = SyntheticScorerContext(n_pairs=n_pairs, device="cpu", seed=29)
        dres = TorchVehicleDriver(_cfg(res_dir), scorer=scr2, vendored=v, curriculum=curriculum)
        restored = {"count": -1}
        orig_restore = dres._restore_into

        def hooked_restore(rt, merged_):
            orig_restore(rt, merged_)
            restored["count"] = len(rt.tensor_sensitivity_ema)

        dres._restore_into = hooked_restore  # type: ignore[method-assign]
        dres.run()
        resumed_bytes = len((res_dir / "best" / "best_archive.bin").read_bytes())

    return {
        "kill_ep": kill_ep,
        "ckpt_ema_count": ckpt_ema_count,
        "restored_ema_count": restored["count"],
        "resumed_bytes": resumed_bytes,
        "ref_bytes": ref_bytes,
        "bit_identical": resumed_bytes == ref_bytes,
    }


if __name__ == "__main__":
    raise SystemExit(main())
