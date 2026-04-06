# PRD — smarter segmentation main ROI

## Goal
Build and measure a dynamic, compression-side ROI experiment that keeps the main central ROI protected, adds an auxiliary ROI only when persistent evidence supports it, and tests whether this can beat the current authoritative 3.33 floor.

## Non-goals
- No decoder-side learned model in this phase.
- No external-weight dependency in `inflate.sh`.
- No broad rewrite of the submission path.

## User-visible success
- The repo contains a concrete, execution-ready experiment for dynamic main-ROI compression.
- The experiment leaves honest artifacts and can be resumed by a fresh loop.
- The main ROI remains explicit and is not replaced by only peripheral saliency.

## Constraints
- Use `uv` for Python execution and package management.
- Keep local CPU scoring authoritative.
- Keep `current_workflow` and `rule_faithful` reporting separate.
- Stay inside the existing mutation frontier.
