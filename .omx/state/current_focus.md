# Current Focus — 2026-04-10 00:15 CDT

## Floor
- **Promoted honest floor**: `1.557` (hardened proxy, h=64 saliency-fixed, ep 1098)
- **Previous floor**: `1.727` (h=32 QAT+EMA)
- **Public target to beat**: `1.89` (neural_inflate PR#49)

## Active training
- `standard_h64_long2500_v2` — live on MPS
  - ep 11, scorer 1.4215 (hardened hard-argmax evaluator)
  - improving every epoch, still in warmup
  - PID 89412, PYTHONUNBUFFERED=1
  - crash-proof: atomic saves + signal handlers + atexit

## tac v0.7.0 hardening (this session)
- Hard argmax SegNet in checkpoint selection (was soft cosine — wrong metric)
- Hard argmax SegNet in proxy scorer (matches official evaluate.py)
- Atomic writes for all saves (tmp + rename on POSIX)
- Signal handlers (SIGTERM/SIGINT/SIGHUP) + atexit for emergency saves
- Pydantic TrainConfig with 12 validators
- 54 tests passing, ruff clean, hypothesis fuzzing
- Bugs fixed: boundary mask missing T-dim, boundary mask missing preprocess_input,
  inflate_postfilter missing "standard" variant, save_int8 zero-guard,
  eval double-quantization dedup, EMA device safety on resume

## Next moves
1. Let v2 training run — hardened evaluator may find better checkpoints
2. h=96 on cloud GPU (Modal A10G) — width scaling predicts ~1.45
3. Temperature-annealed loss mode — boundary mask bug now fixed, path viable
4. Prepare submission PR when best score is locked
5. Update writeup with hardening narrative for best-writeup prize
