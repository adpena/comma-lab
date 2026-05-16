#!/usr/bin/env bash
# Apply OR1 META-ASSUMPTION ADVERSARIAL REVIEW axis update to the codex
# plugin adversarial-review skill files. Re-run after any codex plugin
# upgrade that overwrites cache mirrors.
#
# Source of truth: marketplace canonical at
#   ~/.claude/plugins/marketplaces/openai-codex/plugins/codex/{prompts,commands}/adversarial-review.md
#
# This script ensures the cache mirrors at
#   ~/.claude/plugins/cache/openai-codex/codex/<version>/{prompts,commands}/adversarial-review.md
# match the marketplace canonical (which carries the OR1 update).
#
# Per `feedback_or1_codex_review_skill_prompt_assumption_challenge_axis_landed_20260515.md`.
# Lane: lane_or1_codex_skill_prompt_update_20260515.

set -euo pipefail

CANONICAL_PROMPT="${HOME}/.claude/plugins/marketplaces/openai-codex/plugins/codex/prompts/adversarial-review.md"
CANONICAL_COMMAND="${HOME}/.claude/plugins/marketplaces/openai-codex/plugins/codex/commands/adversarial-review.md"

if [[ ! -f "${CANONICAL_PROMPT}" ]]; then
  echo "FATAL: marketplace canonical prompt not found at ${CANONICAL_PROMPT}" >&2
  exit 1
fi
if [[ ! -f "${CANONICAL_COMMAND}" ]]; then
  echo "FATAL: marketplace canonical command not found at ${CANONICAL_COMMAND}" >&2
  exit 1
fi

# Verify the canonical files carry the OR1 update signatures (the
# <meta_assumption_review> block in the prompt + the CARGO-CULTED
# classification language in the command).
if ! grep -q "meta_assumption_review" "${CANONICAL_PROMPT}"; then
  echo "FATAL: marketplace canonical prompt missing <meta_assumption_review> block" >&2
  echo "       Re-apply OR1 update to marketplace canonical first." >&2
  exit 2
fi
if ! grep -q "CARGO-CULTED" "${CANONICAL_COMMAND}"; then
  echo "FATAL: marketplace canonical command missing CARGO-CULTED framing" >&2
  echo "       Re-apply OR1 update to marketplace canonical first." >&2
  exit 2
fi

CACHE_ROOT="${HOME}/.claude/plugins/cache/openai-codex/codex"
if [[ ! -d "${CACHE_ROOT}" ]]; then
  echo "WARN: no codex cache directory at ${CACHE_ROOT}; nothing to sync." >&2
  exit 0
fi

SYNCED=0
for VERSION_DIR in "${CACHE_ROOT}"/*/; do
  [[ -d "${VERSION_DIR}" ]] || continue
  CACHE_PROMPT="${VERSION_DIR}prompts/adversarial-review.md"
  CACHE_COMMAND="${VERSION_DIR}commands/adversarial-review.md"
  if [[ -f "${CACHE_PROMPT}" ]]; then
    cp "${CANONICAL_PROMPT}" "${CACHE_PROMPT}"
    echo "[synced] ${CACHE_PROMPT}"
    SYNCED=$((SYNCED + 1))
  fi
  if [[ -f "${CACHE_COMMAND}" ]]; then
    cp "${CANONICAL_COMMAND}" "${CACHE_COMMAND}"
    echo "[synced] ${CACHE_COMMAND}"
    SYNCED=$((SYNCED + 1))
  fi
done

echo "[apply_codex_adversarial_review_skill_update] ${SYNCED} file(s) synced."
