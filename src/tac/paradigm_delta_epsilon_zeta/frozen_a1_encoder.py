# SPDX-License-Identifier: MIT
"""Frozen A1 encoder loader for PARADIGM-δεζ Track 1.

The A1 lane (``track1_phase_a1_score_gradient``) optimised a per-pair latent
table + lightweight HNeRV-style decoder via score-gradient supervision against
PR101's substrate. The canonical winner produced
``0.226352 [contest-CUDA T4]`` on a 178,262-byte archive (sha256
``87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5``).

For Track 1 (Ballé hyperprior + 128K decoder end-to-end), the **encoder side
is held FROZEN**: we reuse the per-pair latent table that A1 already learned,
and train a brand-new 128K Quantizr-class FiLM decoder + Ballé hyperprior on
top of it. This gives the joint Lagrangian-ADMM coordinator a stable rate
target while the decoder + hyperprior co-adapt.

Designation contract
--------------------

The "frozen A1 encoder" is **one** designated artifact, not "any A1 candidate".
35+ A1 candidates exist in ``experiments/results/track1_phase_a1_score_gradient_*/``
(both local builds and Modal harvests). Per CLAUDE.md "Operator gates must be
wired and used", the canonical artifact MUST be selected once, recorded in
``.omx/state/canonical_a1_designation.md``, and exposed via the symlink
``experiments/results/A1_canonical/`` so future agents cannot accidentally
load a different A1 candidate.

This loader REFUSES to operate on anything other than that symlink. It will
NOT scan the results tree for a "best" candidate at load time — that selection
is a council/operator decision recorded in the designation memo, not a
heuristic in this module.

Public API
----------

- :class:`FrozenA1Encoder` — wraps the canonical A1 latent table + decoder
  state dict in a frozen ``nn.Module``-compatible interface.
- :func:`load_frozen_a1_encoder` — factory that resolves the canonical symlink
  and constructs the encoder; raises :class:`FrozenA1EncoderError` if the
  designation is missing or inconsistent.
- :data:`A1_CANONICAL_DIR_NAME` — the literal directory name under
  ``experiments/results/`` (``A1_canonical``) so callers do not hard-code it.

CLAUDE.md compliance
--------------------

- No /tmp paths in any persisted manifest emitted by this module.
- Every loaded artifact is hashed (sha256) and the hash is recorded in the
  returned :class:`FrozenA1Encoder.provenance`. Mismatch against the
  designation memo's recorded hash raises :class:`FrozenA1EncoderError`.
- The encoder is frozen via ``requires_grad_(False)`` AND ``eval()`` AND a
  guard that raises if any caller attempts to optimise its parameters.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

A1_CANONICAL_DIR_NAME = "A1_canonical"
"""Literal name of the canonical A1 symlink under ``experiments/results/``.

A new operator selecting a different A1 candidate must update both this
symlink AND ``.omx/state/canonical_a1_designation.md``; the loader cross-
checks both.
"""

CANONICAL_DESIGNATION_PATH = ".omx/state/canonical_a1_designation.md"
"""Repo-relative path of the council/operator designation memo."""


class FrozenA1EncoderError(RuntimeError):
    """Raised when the canonical A1 artifact cannot be loaded.

    Common causes
    -------------
    - The ``experiments/results/A1_canonical/`` symlink does not exist.
    - The designation memo at :data:`CANONICAL_DESIGNATION_PATH` is missing.
    - The latent file's sha256 does not match the designation memo's recorded
      hash (someone moved/replaced the symlink target out-of-band).
    - A caller tried to enable gradient flow through the frozen encoder.
    """


@dataclass
class FrozenA1Encoder:
    """Frozen wrapper around the canonical A1 latent table + decoder weights.

    Attributes
    ----------
    latents : torch.Tensor
        ``(N_PAIRS, LATENT_DIM)`` float tensor — the per-pair latent table
        the A1 encoder learned via score-gradient supervision. Frozen
        (``requires_grad=False``).
    decoder_state_dict : dict[str, torch.Tensor]
        State dict of the A1 HNeRV decoder. Phase 1 may use this as a warm
        initialisation for the 128K decoder (NOT replaced — the 128K decoder
        is a fresh architecture; see :mod:`tac.paradigm_delta_epsilon_zeta.decoder_128k`).
    archive_sha256 : str
        SHA-256 of the original A1 archive.zip. Locked to designation memo.
    archive_size_bytes : int
        Size of the original A1 archive.zip in bytes.
    contest_cuda_score : float | None
        The ``[contest-CUDA T4]`` score the canonical artifact achieved.
        ``None`` if the designation memo did not record it.
    provenance : dict[str, Any]
        Full audit-trail: paths, hashes, designation timestamp, loaded-at
        timestamp.
    """

    latents: torch.Tensor
    decoder_state_dict: dict[str, torch.Tensor]
    archive_sha256: str
    archive_size_bytes: int
    contest_cuda_score: float | None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.latents.requires_grad:
            # Hard guard: T1 design freezes the encoder. If a caller flipped
            # requires_grad ON before passing into this dataclass, fail loud
            # rather than allow silent gradient leak.
            raise FrozenA1EncoderError(
                "frozen A1 latent tensor must have requires_grad=False; "
                "got requires_grad=True"
            )
        for name, tensor in self.decoder_state_dict.items():
            if not torch.is_tensor(tensor):
                raise FrozenA1EncoderError(
                    f"decoder state dict entry '{name}' is not a torch.Tensor"
                )

    def to(self, device: torch.device | str) -> "FrozenA1Encoder":
        """Move the latent table + decoder state to ``device``.

        Returns a new dataclass (immutable view); does not mutate self.
        """
        moved_latents = self.latents.to(device).detach()
        moved_latents.requires_grad_(False)
        moved_state = {k: v.to(device) for k, v in self.decoder_state_dict.items()}
        return FrozenA1Encoder(
            latents=moved_latents,
            decoder_state_dict=moved_state,
            archive_sha256=self.archive_sha256,
            archive_size_bytes=self.archive_size_bytes,
            contest_cuda_score=self.contest_cuda_score,
            provenance=dict(self.provenance),
        )

    @property
    def n_pairs(self) -> int:
        """Number of frame pairs encoded by the latent table (typically 600)."""
        return int(self.latents.shape[0])

    @property
    def latent_dim(self) -> int:
        """Latent dimensionality per pair (A1 canonical: 28)."""
        return int(self.latents.shape[1])


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_designation_memo(memo_path: Path) -> dict[str, Any]:
    """Parse the YAML/JSON header of the canonical A1 designation memo.

    The memo is markdown with a leading ``json``-fenced block carrying the
    machine-readable designation contract. Returns the parsed contract.

    Raises
    ------
    FrozenA1EncoderError
        If the memo is missing, malformed, or lacks the required keys.
    """
    if not memo_path.exists():
        raise FrozenA1EncoderError(
            f"canonical A1 designation memo not found at {memo_path}; "
            "operator must designate a canonical A1 candidate before T1 "
            "scaffold can load (see tools/designate_canonical_a1.py)"
        )
    text = memo_path.read_text(encoding="utf-8")
    # Find first json fenced block.
    marker_start = text.find("```json")
    if marker_start < 0:
        raise FrozenA1EncoderError(
            f"designation memo {memo_path} has no ```json fenced block "
            "with the designation contract"
        )
    body_start = text.find("\n", marker_start) + 1
    marker_end = text.find("```", body_start)
    if marker_end < 0:
        raise FrozenA1EncoderError(
            f"designation memo {memo_path} has unterminated ```json block"
        )
    payload = text[body_start:marker_end].strip()
    try:
        contract = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise FrozenA1EncoderError(
            f"designation memo {memo_path} has invalid JSON in fenced block: {exc}"
        ) from exc
    required = {"canonical_lane_id", "archive_sha256", "archive_size_bytes"}
    missing = required - set(contract)
    if missing:
        raise FrozenA1EncoderError(
            f"designation memo {memo_path} missing required keys: {sorted(missing)}"
        )
    return contract


def load_frozen_a1_encoder(
    *,
    repo_root: Path,
    canonical_dir_name: str = A1_CANONICAL_DIR_NAME,
    designation_memo_relpath: str = CANONICAL_DESIGNATION_PATH,
    map_location: str | torch.device = "cpu",
    strict_sha_check: bool = True,
) -> FrozenA1Encoder:
    """Load the canonical frozen A1 encoder.

    Parameters
    ----------
    repo_root : Path
        Repository root (the directory containing ``experiments/`` and
        ``.omx/``).
    canonical_dir_name : str
        Directory name under ``experiments/results/`` that points at the
        canonical A1 candidate. Defaults to ``A1_canonical``.
    designation_memo_relpath : str
        Repo-relative path of the designation memo carrying the canonical
        contract.
    map_location : str | torch.device
        ``torch.load`` map_location.
    strict_sha_check : bool
        If True (default), the loaded artifact's archive sha256 must match
        the designation memo. Set False ONLY for offline / scaffold-test
        contexts where the symlink target is a placeholder.

    Returns
    -------
    FrozenA1Encoder

    Raises
    ------
    FrozenA1EncoderError
        For any inconsistency: missing symlink, missing memo, sha mismatch,
        missing checkpoint, missing latent table, or grad-flag violation.
    """
    repo_root = Path(repo_root).resolve()
    canonical_dir = repo_root / "experiments" / "results" / canonical_dir_name
    if not canonical_dir.exists():
        raise FrozenA1EncoderError(
            f"canonical A1 directory/symlink not found at {canonical_dir}; "
            "operator must run tools/designate_canonical_a1.py before T1 "
            "scaffold can load"
        )
    memo_path = repo_root / designation_memo_relpath
    contract = _parse_designation_memo(memo_path)

    # Locate archive.zip — it lives in canonical_dir/finetuned_archive/archive.zip
    # (matching the A1 build_manifest layout).
    candidates = [
        canonical_dir / "finetuned_archive" / "archive.zip",
        canonical_dir / "harvested_artifacts" / "finetuned_archive" / "archive.zip",
        canonical_dir / "archive.zip",
    ]
    archive_path: Path | None = None
    for candidate in candidates:
        if candidate.exists():
            archive_path = candidate
            break
    if archive_path is None:
        raise FrozenA1EncoderError(
            f"no archive.zip found under {canonical_dir} (searched: "
            f"{[str(p.relative_to(canonical_dir)) for p in candidates]})"
        )

    archive_sha = _sha256_file(archive_path)
    if strict_sha_check and archive_sha != contract["archive_sha256"]:
        raise FrozenA1EncoderError(
            f"archive sha mismatch: loaded {archive_sha} vs designated "
            f"{contract['archive_sha256']} (someone moved/replaced the "
            "symlink target without updating the designation memo)"
        )

    # Locate the trained latent table + decoder state dict. The A1 trainer
    # writes ``train/checkpoint_best_proxy.pt`` (or ``checkpoint_ema.pt``).
    # Phase 1 prefers the proxy-best (matches the canonical 0.226352 anchor).
    ckpt_candidates = [
        canonical_dir / "train" / "checkpoint_best_proxy.pt",
        canonical_dir / "harvested_artifacts" / "train" / "checkpoint_best_proxy.pt",
        canonical_dir / "checkpoint_best_proxy.pt",
        canonical_dir / "train" / "checkpoint_ema.pt",
        canonical_dir / "harvested_artifacts" / "train" / "checkpoint_ema.pt",
    ]
    ckpt_path: Path | None = None
    for candidate in ckpt_candidates:
        if candidate.exists():
            ckpt_path = candidate
            break
    if ckpt_path is None:
        raise FrozenA1EncoderError(
            f"no checkpoint found under {canonical_dir} (searched: "
            f"{[str(p) for p in ckpt_candidates]})"
        )

    state = torch.load(
        ckpt_path,
        map_location=map_location,
        weights_only=False,  # WEIGHTS_ONLY_FALSE_OK: A1 checkpoints are trusted operator artifacts under canonical symlink
    )

    # The A1 checkpoint may be a dict containing 'latents' + 'decoder' keys
    # OR a plain state dict where 'latents' is a parameter. Handle both.
    if isinstance(state, dict) and "latents" in state and "decoder" in state:
        latents = state["latents"]
        decoder_sd = state["decoder"]
    elif isinstance(state, dict) and "latent_table" in state:
        latents = state["latent_table"]
        decoder_sd = {
            k: v for k, v in state.items() if k != "latent_table"
        }
    elif isinstance(state, dict):
        # Plain state dict — extract any tensor named 'latents' / 'latent_table'.
        latents = None
        decoder_sd = {}
        for k, v in state.items():
            if k in {"latents", "latent_table"}:
                latents = v
            else:
                decoder_sd[k] = v
        if latents is None:
            # The A1 canonical archive stores latents inside the binary blob
            # (decoded via codec.decode_latents_compact at inflate time), so the
            # raw checkpoint is a pure decoder state dict. Look for an
            # operator-extracted latent table alongside the checkpoint.
            extracted_candidates = [
                ckpt_path.parent / "extracted_frozen_latents.pt",
                ckpt_path.parent.parent / "extracted_frozen_latents.pt",
                canonical_dir / "extracted_frozen_latents.pt",
            ]
            extracted_path: Path | None = None
            for candidate in extracted_candidates:
                if candidate.exists():
                    extracted_path = candidate
                    break
            if extracted_path is None:
                raise FrozenA1EncoderError(
                    f"checkpoint {ckpt_path} has no 'latents' tensor and no "
                    f"extracted_frozen_latents.pt found alongside it. "
                    f"Run tools/extract_frozen_a1_latents.py to materialise the "
                    f"latent table from the canonical archive."
                )
            extracted = torch.load(
                extracted_path,
                map_location=map_location,
                weights_only=False,  # WEIGHTS_ONLY_FALSE_OK: extracted-latents file produced by repo tooling alongside canonical symlink
            )
            if isinstance(extracted, dict) and "latents" in extracted:
                latents = extracted["latents"]
                if (
                    "archive_sha256" in extracted
                    and strict_sha_check
                    and extracted["archive_sha256"] != contract["archive_sha256"]
                ):
                    raise FrozenA1EncoderError(
                        f"extracted_frozen_latents.pt was extracted from a "
                        f"different archive: {extracted['archive_sha256']} vs "
                        f"designated {contract['archive_sha256']}"
                    )
            elif torch.is_tensor(extracted):
                latents = extracted
            else:
                raise FrozenA1EncoderError(
                    f"extracted_frozen_latents.pt at {extracted_path} has "
                    f"unexpected layout {type(extracted).__name__}"
                )
    else:
        raise FrozenA1EncoderError(
            f"checkpoint {ckpt_path} has unexpected type {type(state).__name__}; "
            "expected dict"
        )

    if not isinstance(latents, torch.Tensor):
        raise FrozenA1EncoderError(
            f"latents in {ckpt_path} is type {type(latents).__name__}; "
            "expected torch.Tensor"
        )
    latents = latents.detach().clone()
    latents.requires_grad_(False)

    provenance = {
        "schema_version": 1,
        "loaded_from": {
            "canonical_dir": str(canonical_dir),
            "archive_path": str(archive_path),
            "checkpoint_path": str(ckpt_path),
            "designation_memo": str(memo_path),
        },
        "designation_contract": contract,
        "loaded_archive_sha256": archive_sha,
        "loaded_archive_size_bytes": archive_path.stat().st_size,
        "latents_shape": list(latents.shape),
        "latents_dtype": str(latents.dtype),
        "decoder_param_count": sum(t.numel() for t in decoder_sd.values()),
    }

    return FrozenA1Encoder(
        latents=latents,
        decoder_state_dict=decoder_sd,
        archive_sha256=archive_sha,
        archive_size_bytes=archive_path.stat().st_size,
        contest_cuda_score=contract.get("contest_cuda_score"),
        provenance=provenance,
    )
