"""#205 Phase-3 SEALED capstone config — the canonical triality DEFINITION + the BYTE-IDENTITY gate.

These tests are the executable form of the operator's gate (2026-07-02): the canonical named
config ``sealed_205`` — resolved from the triality (``tac.witness_autoconfig.derive_sealed_205_config``
= the launcher leg; ``tac.witness_dsl.sealed_205_program`` = the DSL leg) — MUST compile to the exact
Phase-3 SEALED §7 argv (``.omx/research/n205_phase3_recursive_adversarial_review_verdict_20260702.md``
§7) BYTE-IDENTICALLY (modulo ``--out-dir``). A mismatch = a different run than the sealed one =
unacceptable. The two triality legs must also AGREE (same flag set + values).

means != ends: this SEALs a MEANS (a launch config). Only a byte-closed n600 exact row < 0.19110
from ``upstream/evaluate.py`` (contest-CPU/CUDA, NEVER MPS) moves the pointer.
"""
from __future__ import annotations

from pathlib import Path

from tac import witness_autoconfig as wac
from tac.witness_dsl import sealed_205_program

_GT = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"

# The canonical Phase-3 SEALED argv (§7 + the R2 OOM-fix flag), out-dir dropped. This is the
# ground-truth token sequence the operator's launch reproduces; the gate is exact equality against it.
# Sourced from the hand-authored §7 oracle launch.sh PLUS the R2 `--verdict-batch 32` OOM-fix flag
# (review 2026-07-02 C4: the §7 argv is the config that OOM-died; the chunk-the-verdict safety flag now
# belongs IN the launched config -> 84 flags / 151 tokens without --out-dir).
SEALED_205_ARGV_NO_OUTDIR = (
    '--gt-cache', 'experiments/results/mlx_fleet_gt_cache/gt_n600.npz', '--num-pairs', '600',
    '--mlx-device', 'gpu', '--seed', '0', '--epochs', '1000', '--eval-every', '25',
    '--verdict-pairs', '0', '--async-verdict', '--verdict-batch', '32', '--curriculum',
    '--tau-softplus-start-epoch',
    '300', '--tau-softplus-tau', '0.3', '--l7-start-epoch', '1000', '--muon-start-epoch',
    '726', '--muon-lr', '0.002', '--muon-momentum', '0.95', '--muon-ns-steps', '5',
    '--stage-transition-rewarmup-epochs', '8', '--stage-transition-rewarmup-floor', '0.1',
    '--stage-transition-rewarmup-shape', 'linear', '--stage-transition-reset-moments',
    '--w-seg', '100', '--w-pose', '1.0', '--score-domain-loss', '--pose-carrier',
    '--pose-carrier-residual-mode', 'table', '--mod-dim', '32', '--hidden-dim', '96',
    '--n-hidden', '4', '--activation', 'hosc', '--hosc-beta', '1.0', '--hosc-beta-end', '4.0',
    '--hosc-beta-anneal', 'linear', '--hosc-omega', '1.0', '--siren-init',
    '--softmax-temp-start', '1.0', '--softmax-temp-end', '0.05', '--tau-anneal-shape',
    'cosine', '--self-orient', '--n-dir-freqs', '2', '--freq-across', '32', '--freq-along',
    '4', '--reorient-every', '50', '--max-bank-freq', '64', '--chroma', '--palette-anchor',
    '--eikonal-weight', '0.01', '--length-weight', '0.001', '--render-h', '384', '--render-w',
    '512', '--render-aa', 'none', '--lane-render-band', '--lane-band-start-epoch', '300',
    '--lane-band-uncertainty-source', 'witness', '--lane-band-tau', '0.85', '--lane-band-eps',
    '0.35', '--lane-band-softness', '1.0', '--lane-band-dash-forward-max-m', '55.0',
    '--lane-band-weight', '1.0', '--persistence-loss-weight', '1.0',
    '--persistence-recall-weight', '1.0', '--cldice-iters', '5', '--persistence-warmup-epochs',
    '300', '--persistence-classes', 'auto', '--amplify-weight', '1.0', '--amplify-form',
    'hinge', '--amplify-margin-target', '1.0', '--amplify-persist', 'inverse_thickness',
    '--island-dilate-px', '1', '--structured-init', '--structured-init-include-lane',
    '--lane-prior-phi1', '--lane-prior-phi1-mode', 'replace', '--lane-prior-phi1-dash-gate',
    '--accum-pairs', '8', '--grad-clip', '1.0', '--ema-decay', '0.997', '--lr', '1e-3',
    '--lr-end', '1e-4', '--weight-decay', '1e-4', '--adam-beta2', '0.999', '--ckpt-every',
    '25', '--stage-checkpoints',
)

_ORACLE_LAUNCH_SH = Path(
    "experiments/results/levelset_n600_witness_capstone_20260702T204138Z/launch.sh")


def _config_argv(cfg, out_dir: str) -> list[str]:
    """Render a WitnessConfig's flags to a flat argv token list (bare flags = single token)."""
    argv: list[str] = []
    for flag, val in cfg.to_trainer_flags(out_dir):
        argv.append(flag)
        if val is not None:
            argv.append(str(val))
    return argv


def _drop_outdir(argv: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--out-dir":
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out


def _sealed_cfg():
    return wac.derive_sealed_205_config(_GT, num_pairs=600, epochs=1000)


# --------------------------------------------------------------------------- THE GATE
def test_sealed_205_argv_byte_identical_to_sealed_section7():
    """THE GATE: the sealed_205 config compiles to the EXACT §7 argv (modulo --out-dir)."""
    cfg = _sealed_cfg()
    got = _drop_outdir(_config_argv(cfg, "/ANY/OUT/DIR"))
    assert tuple(got) == SEALED_205_ARGV_NO_OUTDIR, (
        "sealed_205 argv drifted from the SEALED §7 config — a different run than the sealed one.")


def test_sealed_205_argv_matches_oracle_launch_sh_if_present():
    """Defense-in-depth: match the hand-authored §7 oracle launch.sh (skipped if it was cleaned)."""
    import shlex
    import re
    if not _ORACLE_LAUNCH_SH.exists():
        return  # oracle lives under experiments/results (rebuildable); embedded tuple is authoritative
    joined = _ORACLE_LAUNCH_SH.read_text().replace("\\\n", " ")
    m = re.search(r"train_levelset_witness_realized_through_R_mlx\.py(.*)", joined, re.S)
    assert m, "oracle launch.sh has no trainer invocation"
    oracle = _drop_outdir(shlex.split(m.group(1)))
    assert tuple(oracle) == SEALED_205_ARGV_NO_OUTDIR, "oracle launch.sh differs from embedded §7 argv"


def test_sealed_205_flag_count_and_no_invented_flags():
    """All 83 sealed flags exist in the trainer's real argparse (never-invent-flags / NO-FAKE)."""
    from tools import launch_witness_run as lwr  # noqa: PLC0415
    cfg = _sealed_cfg()
    all_pass, results = lwr.validate_emitted_flags(cfg, "/OUT")
    assert all_pass, f"invented flag(s): {[f for f, ok in results if not ok]}"
    assert len(results) == 84, f"expected 84 flags (83 §7 + R2 --verdict-batch), got {len(results)}"


# --------------------------------------------------------------------------- the 4 SEALED deltas
def test_sealed_205_carries_the_four_sealed_deltas():
    """The 4 deltas over the all-levers base are present with the SEALED values (§2 verdict)."""
    cfg = _sealed_cfg()
    fmap = {f: v for f, v in cfg.to_trainer_flags("/OUT")}
    assert cfg.sealed_205 is True
    assert fmap["--mod-dim"] == 32              # Q4 (over all-levers Whitney floor 19)
    assert str(fmap["--adam-beta2"]) == "0.999"  # Q5 (over all-levers 0.9999999)
    assert str(fmap["--w-pose"]) == "1.0"       # Q2 (over proven_base 0)
    assert "--pose-carrier" in fmap             # Q2 (add)
    assert fmap["--pose-carrier-residual-mode"] == "table"  # Q2 (add)
    # lr trio rendered as the §7 string literals (byte-identity, == trainer float defaults)
    assert fmap["--lr"] == "1e-3" and fmap["--lr-end"] == "1e-4" and fmap["--weight-decay"] == "1e-4"


# --------------------------------------------------------------------------- triality: DSL leg agrees
def test_sealed_205_dsl_program_validates_and_agrees_with_autoconfig():
    """The DSL leg (sealed_205_program) validates and carries the SAME flag set + values."""
    out = "/OUT"
    cfg = _sealed_cfg()
    prog = sealed_205_program(out, _GT, 600, 1000)
    assert prog.validate() == [], f"DSL program violations: {prog.validate()}"

    def norm(v):
        return "<BARE>" if v is None or v is True else str(v)

    ac = {f: norm(v) for f, v in cfg.to_trainer_flags(out)}
    ds = {f: norm(v) for f, v in prog.flag_dict().items()}
    assert ac == ds, (
        f"DSL/autoconfig disagree: ac-only={ {k: ac[k] for k in ac if ac.get(k)!=ds.get(k)} } "
        f"ds-only={ {k: ds[k] for k in ds if ds.get(k)!=ac.get(k)} }")


# --------------------------------------------------------------------------- store_nothing_205 (F6)
def test_store_nothing_205_is_sealed_plus_exactly_one_flag():
    """The STORE-NOTHING variant = sealed_205 + exactly one trailing --pose-carrier-source generated;
    sealed_205 itself stays BYTE-IDENTICAL (never emits --pose-carrier-source)."""
    sealed = _sealed_cfg()
    snn = wac.derive_store_nothing_205_config(_GT, num_pairs=600, epochs=1000)
    assert sealed.pose_carrier_source == "real_keyframe"
    assert snn.pose_carrier_source == "generated"
    sealed_flags = sealed.to_trainer_flags("/OUT")
    snn_flags = snn.to_trainer_flags("/OUT")
    # sealed emits NO --pose-carrier-source (byte-identity preserved); store_nothing emits exactly one.
    assert not any(f == "--pose-carrier-source" for f, _ in sealed_flags)
    extra = [(f, v) for (f, v) in snn_flags if (f, v) not in sealed_flags]
    assert extra == [("--pose-carrier-source", "generated")], f"unexpected delta: {extra}"
    # the store-nothing flag is placed right after the pose SLOT block (--pose-carrier-residual-mode).
    fnames = [f for f, _ in snn_flags]
    assert (fnames.index("--pose-carrier-source")
            == fnames.index("--pose-carrier-residual-mode") + 1)


def test_store_nothing_205_flags_all_valid_in_trainer_argparse():
    """No invented flags: every store_nothing_205 flag (incl. --pose-carrier-source) exists in the
    trainer's real argparse (never-invent-flags / NO-FAKE)."""
    from tools import launch_witness_run as lwr  # noqa: PLC0415
    snn = wac.derive_store_nothing_205_config(_GT, num_pairs=600, epochs=1000)
    ok, results = lwr.validate_emitted_flags(snn, "/OUT")
    assert ok, f"invented flag(s): {[f for f, good in results if not good]}"
    assert len(results) == 85, f"expected 85 flags (84 sealed incl. R2 --verdict-batch + 1 store-nothing), got {len(results)}"


# --------------------------------------------------------------------------- backward-compat
def test_backward_compat_all_levers_and_proven_base_still_render():
    """--all-levers and the proven_base default still derive + flag-validate (no regression)."""
    from tools import launch_witness_run as lwr  # noqa: PLC0415
    for all_levers in (True, False):
        cfg = wac.derive_config(_GT, num_pairs=600, overfit=True, epochs=1000, all_levers=all_levers)
        assert cfg.sealed_205 is False
        ok, results = lwr.validate_emitted_flags(cfg, "/OUT")
        assert ok, f"all_levers={all_levers} emitted invented flag(s)"
        # sanity: sealed-only flags NOT present in the non-sealed configs
        fmap = {f for f, _ in cfg.to_trainer_flags("/OUT")}
        assert "--pose-carrier" not in fmap
