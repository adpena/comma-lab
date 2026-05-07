#!/usr/bin/env bash
# Canonical auto-resume helper for training dispatch scripts.
#
# Per `feedback_grand_council_universal_auto_resume_pattern_20260507.md`
# (council 8/8 ENDORSE), ALL training dispatch scripts that risk preemption
# (Vast.ai / Modal / Lightning) MUST honor a `training_state_<tag>.pt`
# checkpoint if present. This file is the single canonical implementation
# of that pattern; lane scripts source it instead of copy-pasting the
# bash logic.
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline": there
# is ONE canonical helper, sourced everywhere. Adding the auto-resume logic
# to each lane script as inline duplicate bash is FORBIDDEN.
#
# Reference commit (Q-FAITHFUL pattern): da22c989.
#
# USAGE:
#   source scripts/lib_auto_resume.sh
#   AUTO_RESUME_ARGS=()  # array; will be populated by detect_auto_resume
#   detect_auto_resume "$LOG_DIR/train" "training_state_<tag>.pt"
#   "$PYBIN" -u src/tac/experiments/train_renderer.py \
#       --profile <profile> --tag <tag> --device cuda --seed 1234 \
#       --output-dir "$LOG_DIR/train" \
#       "${AUTO_RESUME_ARGS[@]}"   # expands to --resume-from <path> or empty
#
# CONTRACT:
#   - Reads RESUME_FROM env override first; if set, uses it directly.
#   - Otherwise, scans <log_dir>/<filename_pattern> for newest match.
#   - Sets the global AUTO_RESUME_ARGS array (caller must declare it).
#   - Logs the resolution decision via stderr.

# Detect a training_state checkpoint and populate AUTO_RESUME_ARGS array.
#
# Args:
#   $1: log_dir — directory to scan (e.g. "$LOG_DIR/train")
#   $2: filename_pattern — glob to match (e.g. "training_state_q_faithful_modal.pt")
#
# Reads:
#   $RESUME_FROM (optional env var) — explicit override path
#
# Writes:
#   AUTO_RESUME_ARGS (array) — either ("--resume-from" "<path>") or empty
detect_auto_resume() {
    local log_dir="${1:?log_dir required}"
    local fname_pattern="${2:?filename_pattern required}"
    local resume_path=""

    # 1) Env override takes precedence.
    if [ -n "${RESUME_FROM:-}" ]; then
        resume_path="$RESUME_FROM"
        echo "[auto-resume] env RESUME_FROM=$resume_path" >&2
    fi

    # 2) Auto-detect from log_dir.
    if [ -z "$resume_path" ]; then
        # ls -t newest-first; head -1 picks the freshest match. Quote-safe via
        # nullglob fallback if no match.
        local newest
        newest=$(ls -t "$log_dir/$fname_pattern" 2>/dev/null | head -1)
        if [ -n "$newest" ] && [ -f "$newest" ]; then
            resume_path="$newest"
            local sz
            sz=$(stat -c '%s' "$resume_path" 2>/dev/null || stat -f '%z' "$resume_path" 2>/dev/null || echo "?")
            echo "[auto-resume] detected $resume_path ($sz bytes) — preempted run survives" >&2
        fi
    fi

    # 3) Validate + populate args array.
    AUTO_RESUME_ARGS=()
    if [ -n "$resume_path" ]; then
        if [ ! -f "$resume_path" ]; then
            echo "[auto-resume] FATAL: RESUME_FROM=$resume_path is not a file" >&2
            return 3
        fi
        AUTO_RESUME_ARGS=(--resume-from "$resume_path")
        echo "[auto-resume] passing --resume-from $resume_path to train_renderer.py" >&2
    else
        echo "[auto-resume] NO checkpoint detected — starting fresh from epoch 0" >&2
    fi
    return 0
}
