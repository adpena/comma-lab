---
name: DX Must Be Bulletproof
description: Production-hardened DX is non-negotiable. One command, zero ambiguity, validate before running.
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
The eval pipeline failed 5 times due to path issues, missing files, and wrong argument order. This is unacceptable.

**Why:** Every failed attempt wastes 10 minutes of inflate time. With 22 days left, we can't afford debugging DX issues. The pipeline must work first try, every time.

**How to apply:**
- eval.py must validate ALL inputs exist before starting any computation
- All paths resolved to absolute immediately
- inflated/ directory auto-created inside submission_dir (where evaluator expects it)
- archive.zip auto-copied/linked to submission_dir if not present
- Clear error messages with expected vs actual paths
- Click CLI with --help on every script
- One command: `python eval.py --upstream-dir X` does everything
- Pre-flight checks: verify ffmpeg, scorer models, video files all exist
- Use Click for all Python CLIs. No raw sys.argv. No argparse.
