"""apogee_intN candidate generator (concrete plugin).

Mirrors the original `_generator_apogee_intn()` that lived inside
`tools/feedback_loop_sweep.py`, but exposed as a `CandidateGenerator` so the
loop driver does not need to know about apogee. The original function in
`tools/feedback_loop_sweep.py` is preserved verbatim (linter-hardened); this
module is the new canonical location.

Per CLAUDE.md non-negotiables: candidates default to
`ready_for_exact_eval_dispatch=False` with `evidence_semantics="byte_only_forensic"`,
matching the original. Promotion to dispatch requires exact-SHA non-proxy
evidence which lives outside this generator.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from tac.optimizer.sweep_plugin import (
    Candidate,
    CandidateGenerator,
    DispatchSpec,
    register_generator,
)

REPO = Path(__file__).resolve().parents[4]


class ApogeeIntNGenerator(CandidateGenerator):
    """Walks `experiments/results/apogee_int<N>_repack_*` for ready archives."""

    name = "apogee_intN"

    DEFAULT_LANE_SCRIPT = "scripts/remote_lane_apogee_intN.sh"
    SUPPORTED_NS = (4, 5, 6, 7, 8)

    def __init__(self, repo: Path | None = None) -> None:
        self.repo = (repo or REPO).resolve()

    def __call__(self) -> list[Candidate]:
        candidates: list[Candidate] = []
        for n in self.SUPPORTED_NS:
            repack_dirs = sorted(
                self.repo.glob(f"experiments/results/apogee_int{n}_repack_*")
            )
            if not repack_dirs:
                continue
            repack_dir = repack_dirs[-1]
            meta_path = repack_dir / "repack_metadata.json"
            if not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                continue
            archive_path = repack_dir / f"apogee_int{n}_archive.zip"
            archive_bytes = meta.get("archive_size_bytes")
            rel_err = meta.get("rel_err_pct_per_weight")
            if archive_bytes is None or rel_err is None or not archive_path.is_file():
                continue
            candidates.append({
                "candidate_id": f"apogee_int{n}",
                "archive_bytes": archive_bytes,
                "rel_err_pct": rel_err,
                "n_layers": meta.get("n_intn_layers", 13),
                "lane_class": "apogee_intN",
                "archive_path": archive_path,
                "archive_sha256": meta.get("candidate_archive_sha256"),
                "ready_for_exact_eval_dispatch": False,
                "evidence_semantics": "byte_only_forensic",
                "dispatch_blockers": [
                    "missing_contest_faithful_distortion_model",
                    "missing_exact_sha_non_proxy_readiness_evidence",
                ],
                "score_claim": False,
            })
        return candidates

    def build_dispatch(self, candidate: Candidate, *, label: str) -> DispatchSpec:
        band = candidate.get("predicted_band") or [
            candidate.get("band_low", 0.0),
            candidate.get("band_high", 1.0),
        ]
        cmd = [
            sys.executable,
            str(self.repo / "tools" / "lightning_dispatch_pr106_stack.py"),
            "--lane-script", self.DEFAULT_LANE_SCRIPT,
            "--label", label,
            "--predicted-band", str(band[0]), str(band[1]),
        ]
        if candidate.get("archive_sha256"):
            cmd += ["--expected-archive-sha256", str(candidate["archive_sha256"])]
        if candidate.get("archive_bytes"):
            cmd += ["--expected-archive-size-bytes", str(candidate["archive_bytes"])]
        return DispatchSpec(label=label, cmd=cmd, estimated_cost_usd=0.30,
                            timeout_seconds=1800.0, cwd=self.repo)


register_generator("apogee_intN", ApogeeIntNGenerator)
