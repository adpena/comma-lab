#!/usr/bin/env python3
"""Build a baseline submission archive from canonical components.

Council 2026-04-27 strategic pivot: the saved "baseline_dilated_h64_0_90"
archive uses 48x64 masks (scoring 53.60), NOT the full-res 384x512 masks
that produced the historical 0.9001 record. This script regenerates the
correct full-res masks via SegNet on GT video and packages them with the
existing renderer + poses, producing an archive we can RE-VERIFY on CUDA.

Inputs (defaults match submissions/baseline_dilated_h64_0_90/):
  --renderer  : renderer.bin (ASYM)
  --poses     : optimized_poses.pt (or .bin)
  --gt-video  : upstream/videos/0.mkv
  --crf       : AV1 CRF for masks.mkv (default 50, matches "CRF=50" record)
  --output    : output archive.zip path

Output:
  archive.zip with renderer.bin + masks.mkv + optimized_poses.pt
  Plus a sidecar provenance.json with SHA256s.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "upstream"))

# DETERMINISTIC_ZIP_OK — codex R5-r6 #5: this builder uses the
# deterministic-zip helper below rather than vanilla zipfile.ZipFile.write.
# Two reruns with identical inputs produce byte-identical archive bytes.
_DET_ZIP_DATE_TIME = (2026, 4, 27, 0, 0, 0)
_DET_ZIP_EXTERNAL_ATTR = (0o644 & 0xFFFF) << 16  # rw-r--r--
_DET_ZIP_CREATE_SYSTEM = 3                       # Unix


def _det_zip_write(
    z: zipfile.ZipFile,
    arcname: str,
    src: Path,
    *,
    compress_type: int = zipfile.ZIP_DEFLATED,
    compresslevel: int = 9,
) -> None:
    """Codex R5-r6 #5: write src into z with FIXED metadata.

    Vanilla ``z.write(path, arcname=...)`` embeds the source-file mtime +
    OS-dependent perm bits, so two reruns produce different archive
    bytes. This wrapper forces a constant date_time + Unix perms so the
    archive is reproducible from the same inputs.
    """
    info = zipfile.ZipInfo(filename=arcname, date_time=_DET_ZIP_DATE_TIME)
    info.compress_type = compress_type
    info.external_attr = _DET_ZIP_EXTERNAL_ATTR
    info.create_system = _DET_ZIP_CREATE_SYSTEM
    with open(src, "rb") as f:
        data = f.read()
    z.writestr(info, data, compress_type=compress_type, compresslevel=compresslevel)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--renderer", type=Path,
                   default=REPO / "submissions" / "baseline_dilated_h64_0_90" / "renderer.bin")
    p.add_argument("--poses", type=Path,
                   default=REPO / "submissions" / "baseline_dilated_h64_0_90" / "optimized_poses.pt")
    p.add_argument("--gt-video", type=Path,
                   default=REPO / "upstream" / "videos" / "0.mkv")
    p.add_argument("--crf", type=int, default=50,
                   help="AV1 CRF for masks (default 50, matches '0.9001 + CRF=50' record)")
    p.add_argument("--output", type=Path,
                   default=REPO / "submissions" / "baseline_dilated_h64_0_90"
                   / "archive_rebuilt_full_res.zip")
    p.add_argument("--device", type=str, default=None,
                   help="cpu/mps/cuda for SegNet. DEFAULTS TO CUDA-REQUIRED. "
                        "MPS produces different SegNet outputs per CLAUDE.md "
                        "→ different mask bytes → different score. CUDA is "
                        "the only fully reproducible choice.")
    p.add_argument("--seed", type=int, default=1234,
                   help="Random seed (matches upstream/evaluate.py default).")
    p.add_argument("--half-frame", action="store_true", default=False,
                   help="Quantizr trick: encode only 600 ODD-frame masks "
                        "(frames 1, 3, 5, ..., 1199) instead of 1200. The "
                        "inflate path detects this and warps to recover even "
                        "frames. Roughly halves the mask byte count. Required "
                        "to reach the historical 0.9001 archive size (~338KB).")
    p.add_argument("--with-uniward-delta", type=Path, default=None,
                   help="Path to a Lane C delta.bin produced by "
                        "experiments/optimize_uniward_delta.py. If set, the "
                        "file is bundled into the archive at the canonical "
                        "name 'delta.bin'. The inflate path detects it and "
                        "applies a sparse, L∞-bounded perturbation to the "
                        "rendered frames BEFORE upscale (no scorer at "
                        "inflate — pure additive lookup table). Council "
                        "Lane C target: ≤5KB blob, +0.003 rate cost, "
                        "predicted -0.05 to -0.20 distortion when stacked "
                        "with Lane A pose-TTO. NOTE: by default, a δ marked "
                        "compliance_status=pending_ruling will be REFUSED — "
                        "see --allow-pending-compliance for the explicit "
                        "override.")
    # Codex R5 HIGH — silent contest-noncompliance gate. Lane C δ.bin is a
    # scorer-derived artifact; without an explicit council ruling, bundling
    # it into a submission archive risks shipping a non-compliant entry.
    # We make that risk an EXPLICIT operator decision instead of a silent
    # default. The flag must be passed every time, and is recorded into
    # provenance so audit logs show it.
    p.add_argument("--allow-pending-compliance", action="store_true",
                   default=False,
                   help="Override the pending-ruling refusal for Lane C δ. "
                        "Required to bundle a δ.bin whose header carries "
                        "compliance_status=pending_ruling (the default for "
                        "any newly-built δ). Without this flag the script "
                        "exits non-zero. The override is recorded in the "
                        "provenance JSON so any downstream auditor can see "
                        "that the operator explicitly opted in. ONLY pass "
                        "this for diagnostic / paper-figure builds — never "
                        "for a contest submission until the Yousfi PR #35 "
                        "strict-scorer-rule ruling is in.")
    # Lane B-alt 2026-04-27: brotli-compress renderer.bin into the archive.
    # Inflate side already supports renderer.bin.br auto-decompression
    # (submissions/robust_current/inflate_renderer.py
    # _decompress_brotli_in_archive). Local measurement: q=11 saves ~35KB on
    # the dilated-h64 296KB renderer → ~-0.023 score. Pure rate-side win,
    # contest-compliant under PR #35.
    p.add_argument("--use-brotli", action="store_true", default=False,
                   help="Brotli-compress renderer.bin → renderer.bin.br "
                        "before adding to the archive. Inflate side auto-"
                        "decompresses on extract. Saves ~12%% / ~35KB on "
                        "the 296KB dilated-h64 renderer → -0.023 score "
                        "contribution. (Lane B-alt.)")
    p.add_argument("--brotli-quality", type=int, default=11,
                   help="Brotli quality level 0-11 (default 11 = max, "
                        "matches Quantizr). Higher = smaller archive but "
                        "slower compress. Decompress speed is independent.")
    # Lane M+ 2026-04-27: zero-archive-cost poses computed at inflate from
    # lane-mark mask displacement. The optimized_poses.pt artifact (~7-15KB)
    # is omitted entirely; a 0-byte sentinel ZERO_COST_POSES_SENTINEL is
    # written so the inflate side knows to call
    # tac.lane_mark_pose.compute_zero_cost_poses_from_masks() instead of
    # erroring out on missing poses. Inflate side requires
    # INFLATE_ZERO_COST_POSES=1 to actually compute (env gate prevents
    # silent activation on stale archives).
    #
    # Net rate impact: -7-15KB → roughly -0.005 score contribution.
    # Distortion impact: untested vs baseline; council Lane A variant.
    p.add_argument("--use-zero-cost-poses", action="store_true", default=False,
                   help="Lane M+: omit optimized_poses.pt from the archive "
                        "and write a zero_cost_poses_v1 sentinel instead. "
                        "Inflate side computes per-pair 6-DOF poses from "
                        "lane-mark mask displacement (zero archive bytes). "
                        "Requires INFLATE_ZERO_COST_POSES=1 at inflate. "
                        "Saves ~7-15KB → -0.005 score contribution. "
                        "Distortion impact is untested vs baseline; treat "
                        "as a Lane A archive variant. See "
                        "src/tac/lane_mark_pose.py for the analytical math.")
    args = p.parse_args()

    # When --use-zero-cost-poses is set, the poses file is OMITTED from
    # the archive (Lane M+: computed at inflate from lane-mark masks). We
    # still validate the path exists if the operator passed a non-default
    # --poses, so a typo on the CLI is not silently swallowed by the flag.
    _required_inputs = [
        ("renderer", args.renderer),
        ("gt-video", args.gt_video),
    ]
    if not args.use_zero_cost_poses:
        _required_inputs.append(("poses", args.poses))
    for label, path in _required_inputs:
        if not path.exists():
            raise SystemExit(f"--{label} does not exist: {path}")
    # Validate Lane C δ if requested.
    uniward_compliance_status: str | None = None
    uniward_attestation_info: dict | None = None
    if args.with_uniward_delta is not None:
        if not args.with_uniward_delta.exists():
            raise SystemExit(
                f"--with-uniward-delta does not exist: {args.with_uniward_delta}"
            )
        # Codex R5 HIGH fix — silent contest-noncompliance gate. Read the
        # δ.bin header BEFORE we open the SegNet / write the archive so the
        # operator wastes no work on a build that would be refused anyway.
        from tac.uniward_delta import (
            unpack_sparse_delta as _uwd_unpack_for_check,
            COMPLIANCE_PENDING as _UWD_PENDING,
            COMPLIANCE_APPROVED as _UWD_APPROVED,
            COMPLIANCE_REJECTED as _UWD_REJECTED,
        )
        try:
            _uwd_blob = args.with_uniward_delta.read_bytes()
            _uwd_spec_check = _uwd_unpack_for_check(_uwd_blob, device="cpu")
        except Exception as e:  # pragma: no cover — corrupt input
            raise SystemExit(
                f"FATAL: --with-uniward-delta could not be parsed: {e!r}. "
                f"Refusing to bundle an unverifiable δ into the archive."
            )
        uniward_compliance_status = _uwd_spec_check.compliance_status
        if uniward_compliance_status == _UWD_REJECTED:
            raise SystemExit(
                f"FATAL: --with-uniward-delta is marked "
                f"compliance_status={_UWD_REJECTED!r}. This δ has been "
                f"explicitly flagged as non-compliant by the council and "
                f"may NEVER be bundled. Build aborted."
            )
        if uniward_compliance_status == _UWD_PENDING and not args.allow_pending_compliance:
            raise SystemExit(
                "FATAL: --with-uniward-delta is marked "
                f"compliance_status={_UWD_PENDING!r}. Lane C δ.bin is a "
                "SCORER-DERIVED artifact; Yousfi PR #35 strict-scorer-rule "
                "may class this as non-compliant. Refusing to bundle.\n"
                "  To proceed for a diagnostic / paper-figure build only, "
                "pass --allow-pending-compliance — the override is recorded "
                "in the archive provenance JSON for audit. DO NOT use this "
                "for a contest submission until the council ruling lands in "
                ".omx/research/findings.md."
            )
        if uniward_compliance_status == _UWD_APPROVED:
            # CODEX R5-2 #4 fix (2026-04-27): operator-self-asserted
            # approval is no longer trusted. The δ.bin header alone is
            # not sufficient to bundle as approved — we ALSO require an
            # external attestation file at the canonical path
            # .omx/state/lane_c_compliance_attestations/<sha>.json,
            # SHA-keyed against the actual δ.bin bytes. This means an
            # operator cannot bypass the gate by editing the header or
            # by issuing --compliance-status approved (the optimizer no
            # longer accepts that value anyway). The attestation must
            # be produced by tools/sign_lane_c_compliance.py — which
            # records the approver identity, ruling text, timestamp,
            # whoami, and git HEAD into the JSON. The verifier here
            # cross-checks attestation.delta_sha256 vs sha256(blob) to
            # catch any drift between the δ that was approved and the
            # δ that ships.
            from tac.lane_c_compliance import (
                verify_attestation_for_blob, AttestationMissing,
                AttestationMismatch, AttestationMalformed,
                AttestationSignatureInvalid,
                AttestationApproverNotInTrustRoot,
                TrustRootMissing, TrustRootMalformed,
                attestation_path_for, compute_blob_sha256,
            )
            _uwd_sha = compute_blob_sha256(_uwd_blob)
            _att_path = attestation_path_for(_uwd_sha, root=REPO)
            try:
                _attestation = verify_attestation_for_blob(
                    _uwd_blob, root=REPO,
                )
            except AttestationMissing:
                raise SystemExit(
                    f"FATAL: --with-uniward-delta is marked "
                    f"compliance_status={_UWD_APPROVED!r} but NO "
                    f"attestation exists at the canonical path:\n"
                    f"  {_att_path}\n"
                    f"  δ.bin sha256: {_uwd_sha}\n\n"
                    "Approval requires an external attestation file "
                    "produced by:\n"
                    "  python tools/sign_lane_c_compliance.py \\\n"
                    f"      --delta-bin {args.with_uniward_delta} \\\n"
                    "      --approver <yousfi|council|fridrich|...> \\\n"
                    "      --ruling-text \"<PR #35 ruling URL or text>\" \\\n"
                    "      --private-key ~/.config/pact/lane_c_signing_key.pem\n\n"
                    "The bare δ.bin header is OPERATOR-CONTROLLED and "
                    "thus untrusted; the attestation is the trust anchor. "
                    "(Codex R5-2 #4 + R5-3 #1)"
                )
            except AttestationMismatch as e:
                raise SystemExit(
                    "FATAL: --with-uniward-delta has compliance_status="
                    f"{_UWD_APPROVED!r} but the attestation at the "
                    "canonical path is for a DIFFERENT δ.bin:\n"
                    f"  {e}\n"
                    "Either the δ was re-built after signing (re-run "
                    "tools/sign_lane_c_compliance.py against the current "
                    "δ.bin) or the wrong attestation was placed at the "
                    "canonical path. (Codex R5-2 #4)"
                )
            except AttestationMalformed as e:
                raise SystemExit(
                    "FATAL: --with-uniward-delta has compliance_status="
                    f"{_UWD_APPROVED!r} but the attestation file is "
                    "malformed:\n"
                    f"  {e}\n"
                    "Re-sign with tools/sign_lane_c_compliance.py "
                    "providing a non-empty --ruling-text and "
                    "--approver and --private-key. (Codex R5-2 #4)"
                )
            except TrustRootMissing as e:
                raise SystemExit(
                    "FATAL: --with-uniward-delta has compliance_status="
                    f"{_UWD_APPROVED!r} but the trust root pubkey "
                    "registry is missing:\n"
                    f"  {e}\n"
                    "Bootstrap the trust root via "
                    "'python tools/lane_c_keygen.py --approver-id <id>' "
                    "and paste the pubkey hex into the registry file. "
                    "(Codex R5-3 #1)"
                )
            except TrustRootMalformed as e:
                raise SystemExit(
                    "FATAL: --with-uniward-delta has compliance_status="
                    f"{_UWD_APPROVED!r} but the trust root pubkey "
                    "registry is malformed:\n"
                    f"  {e}\n"
                    "Fix the JSON or restore from git history. "
                    "(Codex R5-3 #1)"
                )
            except AttestationApproverNotInTrustRoot as e:
                raise SystemExit(
                    "FATAL: --with-uniward-delta has compliance_status="
                    f"{_UWD_APPROVED!r} but the attestation's approver "
                    "is not in the trust root:\n"
                    f"  {e}\n"
                    "Only allowlisted approvers (council members whose "
                    "pubkeys are committed in trust_root_pubkeys.json) "
                    "can issue Lane C compliance attestations. "
                    "(Codex R5-3 #1)"
                )
            except AttestationSignatureInvalid as e:
                raise SystemExit(
                    "FATAL: --with-uniward-delta has compliance_status="
                    f"{_UWD_APPROVED!r} but the Ed25519 signature on "
                    "the attestation FAILED to verify:\n"
                    f"  {e}\n"
                    "The attestation has been tampered with (ruling "
                    "text or approver swapped after signing) OR was not "
                    "signed by the registered key holder. The δ MAY NOT "
                    "be bundled. (Codex R5-3 #1)"
                )
            # Codex R5-3 #5 fix (2026-04-27): provenance now records the
            # FULL attestation record (so an offline auditor can re-run
            # canonical_signed_payload + verify the signature without
            # touching .omx) plus the on-disk attestation file SHA (so
            # file-level integrity is captured) plus an explicit
            # ``delta_sha256`` field (the previous ``attestation_sha256``
            # was misnamed — it was the δ hash, not the file hash).
            try:
                _att_record = json.loads(_att_path.read_text())
            except (OSError, json.JSONDecodeError) as e:  # pragma: no cover — load_attestation already passed
                raise SystemExit(
                    f"FATAL: failed to re-read attestation at {_att_path} "
                    f"for provenance recording: {e!r}. The attestation "
                    "passed verification; this is a filesystem race or "
                    "permissions change between verify and provenance read."
                )
            _att_file_bytes = _att_path.read_bytes()
            _att_file_sha = hashlib.sha256(_att_file_bytes).hexdigest()
            uniward_attestation_info = {
                "attestation_path": str(_att_path),
                # Field rename (Codex R5-3 #5): delta_sha256 is the SHA of
                # the δ.bin bytes; attestation_file_sha256 is the SHA of
                # the on-disk attestation JSON. Backward-compat alias
                # kept for one release so existing audit tooling does not
                # silently break.
                "delta_sha256": _attestation.delta_sha256,
                "attestation_sha256": _attestation.delta_sha256,  # DEPRECATED alias
                "attestation_file_sha256": _att_file_sha,
                "attestation_schema_version": _attestation.schema_version,
                "attestation_approver": _attestation.approver,
                "attestation_signed_at_utc": _attestation.signed_at_utc,
                "attestation_signed_by_user": _attestation.signed_by_user,
                "attestation_git_head": _attestation.git_head,
                "attestation_ruling_text": _attestation.ruling_text,
                "attestation_signature_hex": _attestation.signature_hex,
                "attestation_delta_size_bytes":
                    _attestation.delta_size_bytes,
                "attestation_delta_path_at_signing":
                    _attestation.delta_path_at_signing,
                # Full record (sorted-keys JSON) so an offline auditor can
                # reconstruct canonical_signed_payload byte-exactly and
                # verify the signature without needing the original file.
                "attestation_record": _att_record,
            }
            print(
                f"[build] Lane C δ compliance_status={_UWD_APPROVED!r} "
                f"VERIFIED against external attestation:\n"
                f"  attestation: {_att_path}\n"
                f"  approver:    {_attestation.approver}\n"
                f"  signed at:   {_attestation.signed_at_utc}\n"
                f"  signed by:   {_attestation.signed_by_user}\n"
                f"  git HEAD:    {_attestation.git_head}\n"
                f"  δ.bin sha:   {_uwd_sha}",
            )
        elif args.allow_pending_compliance:
            print(
                f"\n{'=' * 78}\n"
                f"[build] WARNING: bundling δ with compliance_status="
                f"{uniward_compliance_status!r} via "
                f"--allow-pending-compliance override.\n"
                f"  This archive is NOT contest-compliant until council "
                f"ruling on Yousfi PR #35.\n"
                f"  Tag any score [lane-c-pending-ruling] in the run-log.\n"
                f"{'=' * 78}\n",
            )
    elif args.allow_pending_compliance:
        # Catch the operator-error case: --allow-pending-compliance set but
        # no δ to compliance-gate. Surface it loudly rather than silently no-op.
        print(
            "[build] NOTE: --allow-pending-compliance was passed but "
            "--with-uniward-delta is None — flag has no effect.",
        )

    import os
    import torch
    import av  # noqa: F401  # ensure pyav available
    from tac.scorer import extract_gt_masks, load_scorers
    from tac.mask_codec import encode_masks

    # Determinism non-negotiable per CLAUDE.md. Set BEFORE any cuBLAS call.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # Note: torch.use_deterministic_algorithms(True) would force errors on
    # any nondet op. SegNet inference is no_grad pure-forward so it's safe
    # to skip; documenting here so future readers know this is intentional.

    if args.device is None:
        # Default to CUDA. MPS produces DIFFERENT SegNet outputs than CUDA
        # (per memory feedback_mps_cuda_drift_critical: 2x SegNet drift) —
        # generating masks on MPS would produce a DIFFERENT byte-level
        # archive than one generated on CUDA, breaking deterministic
        # reproducibility against contest CUDA eval.
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            raise SystemExit(
                "FATAL: CUDA not available. Mask generation MUST run on CUDA "
                "for 100% deterministic reproducibility against contest eval. "
                "MPS produces different SegNet outputs (2x drift) → different "
                "masks.mkv bytes → different score. Run this script on a "
                "Vast.ai 4090 OR pass --device cpu and accept that the "
                "rebuilt archive will NOT byte-match a CUDA-built one."
            )
    else:
        device = torch.device(args.device)
        if str(device) == "mps":
            print("[build] WARNING: MPS produces different SegNet outputs "
                  "than CUDA (per feedback_mps_cuda_drift_critical). The "
                  "rebuilt archive bytes will NOT match a CUDA-built one. "
                  "This is acceptable for development smoke-testing only.",
                  file=sys.stderr)
    print(f"[build] device={device}")

    # Stage 1: decode GT video to (H, W, 3) uint8 frames
    print(f"[build] decoding GT video {args.gt_video}")
    t0 = time.monotonic()
    import av as _av
    container = _av.open(str(args.gt_video))
    frames = []
    for frame in container.decode(video=0):
        arr = frame.to_ndarray(format="rgb24")
        frames.append(torch.from_numpy(arr))
    container.close()
    print(f"[build] decoded {len(frames)} frames in {time.monotonic()-t0:.1f}s")

    # Stage 2: SegNet → masks at 384x512
    print("[build] loading SegNet from upstream/models/")
    _, segnet = load_scorers(
        posenet_path=REPO / "upstream" / "models" / "posenet.safetensors",
        segnet_path=REPO / "upstream" / "models" / "segnet.safetensors",
        device=device,
        upstream_dir=REPO / "upstream",
    )
    segnet.eval()
    print("[build] extracting masks at 384x512 via SegNet")
    t0 = time.monotonic()
    masks = extract_gt_masks(frames, segnet, device, batch_size=8)
    print(f"[build] masks shape={tuple(masks.shape)} in {time.monotonic()-t0:.1f}s")

    # Half-frame mode: keep only ODD-indexed frames (1, 3, 5, ..., 1199).
    # The inflate side detects 600-mask input and reconstructs even frames
    # via zoom-flow warp (see submissions/robust_current/inflate_renderer.py
    # _expand_half_frame_masks). This is the Quantizr trick — roughly halves
    # the mask byte count without quality loss because the renderer's motion
    # predictor handles the t -> t+1 warp anyway.
    if args.half_frame:
        masks = masks[1::2]  # frames 1, 3, 5, ..., 1199 → 600 frames
        print(f"[build] HALF-FRAME mode: kept odd-indexed only, shape={tuple(masks.shape)}")

    # Stage 3: encode masks as AV1 at requested CRF
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        masks_path = td_path / "masks.mkv"
        print(f"[build] encoding masks at CRF={args.crf}")
        t0 = time.monotonic()
        size = encode_masks(masks, masks_path, crf=args.crf, fps=20)
        print(f"[build] masks.mkv = {size:,} bytes in {time.monotonic()-t0:.1f}s")

        # Lane B-alt 2026-04-27: optional brotli-compress renderer.bin → .br.
        # Done in Stage 3.5 (in the same TemporaryDirectory) so the .br file
        # is gone after the archive is sealed; only the renderer.bin source
        # path persists. Inflate side auto-decompresses .br files.
        renderer_arcname = "renderer.bin"
        renderer_src_for_zip = args.renderer
        renderer_compressed_size: int | None = None
        if args.use_brotli:
            from tac.submission_archive import compress_file_brotli
            renderer_br_path = td_path / "renderer.bin.br"
            compress_file_brotli(
                args.renderer, renderer_br_path,
                quality=args.brotli_quality,
            )
            renderer_compressed_size = renderer_br_path.stat().st_size
            renderer_arcname = "renderer.bin.br"
            renderer_src_for_zip = renderer_br_path
            saved = args.renderer.stat().st_size - renderer_compressed_size
            print(
                f"[build] Lane B-alt brotli q={args.brotli_quality}: "
                f"renderer.bin {args.renderer.stat().st_size:,} → .br "
                f"{renderer_compressed_size:,} bytes (saved {saved:,}, "
                f"-{saved / 37545489 * 25:.4f} score contribution)"
            )

        # Stage 4: build archive.zip
        # Lane M+ (--use-zero-cost-poses): skip optimized_poses.pt entirely
        # and write a 0-byte sentinel ZERO_COST_POSES_SENTINEL so the
        # inflate side can switch to the analytical path. The sentinel
        # filename + the env-gate (INFLATE_ZERO_COST_POSES=1) together
        # ensure no silent activation on a stale archive.
        from tac.lane_mark_pose import ZERO_COST_POSES_SENTINEL
        sentinel_path: Path | None = None
        if args.use_zero_cost_poses:
            sentinel_path = td_path / ZERO_COST_POSES_SENTINEL
            sentinel_path.write_bytes(b"")  # 0-byte marker

        args.output.parent.mkdir(parents=True, exist_ok=True)
        # codex R5-r6 #5: collect entries in deterministic (alphabetical)
        # order, then write via the fixed-timestamp helper so reruns are
        # byte-identical. Order also matters for archive reproducibility.
        det_entries: list[tuple[str, Path]] = [
            (renderer_arcname, renderer_src_for_zip),
            ("masks.mkv", masks_path),
        ]
        if args.use_zero_cost_poses:
            det_entries.append((ZERO_COST_POSES_SENTINEL, sentinel_path))
        else:
            det_entries.append(("optimized_poses.pt", args.poses))
        if args.with_uniward_delta is not None:
            det_entries.append(("delta.bin", args.with_uniward_delta))
        det_entries.sort(key=lambda kv: kv[0])
        with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for arcname, src in det_entries:
                _det_zip_write(z, arcname, src)
        if args.use_zero_cost_poses:
            print(
                f"[build] Lane M+ ZERO-COST POSES: omitted "
                f"optimized_poses.pt ({args.poses.stat().st_size:,} "
                f"bytes saved); wrote sentinel "
                f"{ZERO_COST_POSES_SENTINEL!r} (0 bytes). Inflate "
                f"requires INFLATE_ZERO_COST_POSES=1."
            )
        if args.with_uniward_delta is not None:
            print(f"[build] bundled UNIWARD δ from {args.with_uniward_delta} "
                  f"({args.with_uniward_delta.stat().st_size:,} bytes)")
        archive_size = args.output.stat().st_size
        rate_unscaled = archive_size / 37545489
        rate_contribution = 25 * rate_unscaled
        print("\n=== Archive built ===")
        print(f"  Path: {args.output}")
        print(f"  Bytes: {archive_size:,}")
        print(f"  Rate (unscaled): {rate_unscaled:.6f}")
        print(f"  Rate (score contribution): {rate_contribution:.4f}")

        # Provenance — every input that affects the output bytes is
        # SHA-pinned so a future re-run can detect any drift.
        segnet_path = REPO / "upstream" / "models" / "segnet.safetensors"
        gpu_model = None
        if torch.cuda.is_available():
            try:
                gpu_model = torch.cuda.get_device_name(0)
            except Exception:
                pass
        prov = {
            "schema_version": 1,
            "tool": "experiments/build_baseline_archive.py",
            "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device": str(device),
            "gpu_model": gpu_model,
            "torch_version": torch.__version__,
            "cuda_version": getattr(torch.version, "cuda", None),
            "crf": args.crf,
            "seed": args.seed,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "archive_path": str(args.output),
            "archive_size_bytes": archive_size,
            "archive_sha256": _sha256(args.output),
            "rate_unscaled": rate_unscaled,
            "rate_score_contribution": rate_contribution,
            "segnet_weights_sha256": _sha256(segnet_path) if segnet_path.exists() else None,
            "use_brotli": bool(args.use_brotli),
            "brotli_quality": int(args.brotli_quality) if args.use_brotli else None,
            "use_zero_cost_poses": bool(args.use_zero_cost_poses),
            "components": {
                renderer_arcname: {
                    "source": str(args.renderer),
                    "size_bytes_uncompressed": args.renderer.stat().st_size,
                    "size_bytes_in_archive": (
                        renderer_compressed_size
                        if renderer_compressed_size is not None
                        else args.renderer.stat().st_size
                    ),
                    "sha256_uncompressed": _sha256(args.renderer),
                    "compression": (
                        f"brotli-q{args.brotli_quality}"
                        if args.use_brotli else "none"
                    ),
                },
                "masks.mkv": {
                    "source": "rebuilt from GT via SegNet at 384x512 CRF=50",
                    "size_bytes": size,
                    "sha256": _sha256(masks_path),
                },
                **({
                    ZERO_COST_POSES_SENTINEL: {
                        "source": "lane-mark mask displacement (computed at inflate)",
                        "size_bytes": 0,
                        "sha256": (
                            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934"
                            "ca495991b7852b855"  # SHA256 of empty bytes
                        ),
                        "kind": "lane_m_plus_zero_cost_pose_sentinel",
                        "note": (
                            "Inflate computes per-pair 6-DOF poses via "
                            "tac.lane_mark_pose.compute_zero_cost_poses_from_masks. "
                            "Requires env INFLATE_ZERO_COST_POSES=1."
                        ),
                    }
                } if args.use_zero_cost_poses else {
                    "optimized_poses.pt": {
                        "source": str(args.poses),
                        "size_bytes": args.poses.stat().st_size,
                        "sha256": _sha256(args.poses),
                    },
                }),
                **({"delta.bin": {
                    "source": str(args.with_uniward_delta),
                    "size_bytes": args.with_uniward_delta.stat().st_size,
                    "sha256": _sha256(args.with_uniward_delta),
                    "kind": "lane_c_uniward_sparse_delta",
                    # Codex R5 HIGH fix — record the compliance gate state
                    # so any future auditor reading provenance.json can see
                    # whether this build went through the pending-ruling
                    # override path. The flag MUST appear in the audit
                    # trail; CLAUDE.md "Strategic Secrecy" rule means we
                    # never ship a non-compliant archive without showing
                    # exactly when and how the override was approved.
                    "compliance_status": uniward_compliance_status,
                    "allow_pending_compliance_override": bool(
                        args.allow_pending_compliance
                    ),
                    # CODEX R5-2 #4 fix — for approved δ, record the full
                    # attestation chain in provenance so a future auditor
                    # can reconstruct who approved the ship without
                    # needing to read the attestation JSON separately.
                    # None for pending/rejected δ — the field's presence
                    # IS the audit signal.
                    "external_attestation": uniward_attestation_info,
                }} if args.with_uniward_delta is not None else {}),
            },
        }
        prov_path = args.output.with_suffix(".provenance.json")
        with open(prov_path, "w") as f:
            json.dump(prov, f, indent=2)
        print(f"  Provenance: {prov_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
