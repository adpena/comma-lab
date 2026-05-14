# Layered operator directive — DP1 Comma2k19 streamer 2026-05-14

**Lane**: `lane_dp1_comma2k19_autoload_log_incremental_20260514`
**Active subagent**: `a88b211763b64ed74` (streamer spec)
**Halted subagent**: `adcef0841b60aac5b` (original cache/download spec, superseded)
**Tag**: `research_only=true`; canonical persistence handoff per CLAUDE.md "Subagent coherence-by-default" mandatory pre-read of `.omx/research/*_directive_*` files dated within last 24 hours.

## Spec evolution

1. **Original** (subagent `adcef0841b60aac5b`): download Comma2k19 chunks locally + LRU cache
2. **Pivot 1** (operator verbatim): *"or instead of downloading just configure a local streamer and log"* → streaming + JSONL access log (subagent `a88b211763b64ed74` dispatched with this spec)
3. **Pivot 2** (operator verbatim): *"and means of chunking dynamically"*
4. **Pivot 3** (operator verbatim): *"whatever is most efficient and fasteest"*
5. **Pivot 4** (operator verbatim): *"and production hardened OSS"*

## Three additional layered requirements

### A) Dynamic chunking mechanism

`Comma2k19LocalStreamer` must support operator-configurable chunking strategies — not just fixed-size chunks:

```python
@dataclass(frozen=True)
class DynamicChunkingStrategy:
    mode: Literal["frame_range", "motion_class", "entropy", "saliency", "byte_size", "temporal_window"]
    frame_range_size: int | None = None
    motion_threshold: float | None = None
    entropy_threshold: float | None = None
    saliency_topk: int | None = None
    byte_size_target: int | None = None
    temporal_window_sec: float | None = None

    def chunk_video(self, video_metadata: dict) -> Iterator[ChunkSpec]:
        """Yields ChunkSpec(chunk_id, frame_range, predicted_bytes, decode_hint)."""

# Usage:
streamer.stream_chunks(
    schedule=LogIncrementalSchedule(...),
    chunking=DynamicChunkingStrategy(mode="motion_class", motion_threshold=0.5),
)
```

Enables score-aware streaming: motion-class chunking prioritizes high-motion segments (PoseNet-informative); saliency chunking prioritizes SegNet-boundary-rich segments.

### B) Maximum efficiency + speed (engineering)

Choose the FASTEST viable streaming mechanism. Document the chosen one with benchmark notes. Candidates ranked:
- **HTTP/2 multi-range** via `httpx[http2]` async client (>>>>>>>>>> HTTP/1.1 connection-per-request)
- **Parallel range fetches** (4-8 concurrent ranges per chunk) — saturates bandwidth
- **Zero-copy decode** via pyav direct-from-URL — avoids intermediate `bytes` copies
- **NVDEC / CUDA video decode** if available on target hardware — GPU-accelerated h.264 decode bypasses CPU bottleneck
- **HTTP Keep-Alive + connection pool** — avoid TCP handshake per chunk
- **Brotli/Snappy in-flight compression** via content-encoding negotiation — smaller bytes-on-wire

Document chosen mechanism + benchmark in module docstring (e.g., *"HTTP/2 multi-range via httpx[http2] at ~50 MB/s; pyav URL decode ~30 fps single-threaded"*).

### C) Production-hardened OSS-grade

Per CLAUDE.md "Beauty, simplicity, and developer experience" + "Public Disclosure Hygiene" + comma.ai MIT-EXCLUSIVE OSS posture (per `feedback_comma_ai_oss_alignment_addendum_landed_20260514.md`):

- **Typed API**: frozen dataclasses, `__all__` exports, docstrings with usage examples, runnable doctests where possible
- **Dependency closure**: pin major versions in pyproject.toml; `httpx[http2]>=0.27` + `pyav>=12.0` (both BSD-permissive)
- **License attribution chain**: every streamed chunk carries `license=MIT` + `source_url` + `source_sha256` provenance metadata propagated through codebook (Catalog #210 enforced)
- **No `/tmp` paths** in any persisted artifact (use `<log_dir>/.partial/` for in-flight buffers per CLAUDE.md "Forbidden /tmp paths" non-negotiable)
- **Tests**: 60+ tests minimum (20 streamer + 15 dynamic chunking + 10 feeder + 10 iterator modes + 5 efficiency benchmark mocks)
- **Beautiful runbook docs** at `docs/dp1_comma2k19_streaming_runbook.md` — concrete usage examples, dispatch examples, log-replay-reader API reference
- **Operator audit-friendly**: every dispatch emits JSONL log + one-page `experiments/results/dp1_phase_2_<timestamp>/stream_summary.md`
- **STRICT preflight gate** (claim Catalog # if new bug class): `check_comma2k19_streams_route_through_canonical_streamer` refuses bare `requests.get()` / `httpx.get()` of Comma2k19 URLs outside the canonical helper
- **Catalog #209 + #210 + #211 compliance preserved**
- **CHANGELOG entry** + memory file at canonical path

## Updated deliverables (additive to original 10)

11. `DynamicChunkingStrategy` API with 6+ chunking modes
12. Efficiency benchmark + chosen mechanism documented in module docstring
13. OSS runbook at `docs/dp1_comma2k19_streaming_runbook.md`
14. CHANGELOG entry
15. STRICT preflight gate (Catalog # claim if new bug class)

## Coordination with halted sibling

Subagent `adcef0841b60aac5b` (original cache spec) reported **clean halt** at 2026-05-14T12:46:39Z; no code committed; all `files_touched` records empty; final checkpoint `step: complete, status: complete, notes: superseded by sister a88b211763b64ed74`. Ownership of the lane is `a88b211763b64ed74`.

## Constraints (unchanged + reinforced)

- $0 GPU
- DO NOT stream real chunks in tests (use synthetic-bytes mocks)
- DO NOT auto-stream in this session (operator hasn't dispatched yet)
- NO KILL verdicts
- NO `/tmp` paths
- License attribution mandatory: Comma2k19 = MIT
- Backward compat preserved (synthetic-stub + explicit-chunks-dir modes)
- HNeRV parity discipline 13 lessons
- Honor sister subagents in flight: SIREN, D4-WYNER-ZIV (D1 just landed)
- Commit via `tools/subagent_commit_serializer.py --expected-content-sha256` (Catalog #117 + #157 + #174 + #186)
- Checkpoint via `tools/subagent_checkpoint.py` (Catalog #206)

## Cross-refs

- CLAUDE.md "Subagent coherence-by-default" mandatory pre-flight (this file IS the operator-routed directive subagents read)
- CLAUDE.md "Bugs must be permanently fixed AND self-protected against"
- CLAUDE.md "Beauty, simplicity, and developer experience"
- CLAUDE.md "Public Disclosure Hygiene" + "Forbidden /tmp paths"
- `feedback_comma_ai_oss_alignment_addendum_landed_20260514.md` (MIT-exclusive posture)
- `feedback_pretrained_driving_prior_lane_scaffold_LANDED_20260513.md` (L1 scaffold; license verification chain)
- `feedback_dp1_phase_2_hardening_v2_landed_20260514.md` (Catalog #209/#210/#211 STRICT @ 0)
