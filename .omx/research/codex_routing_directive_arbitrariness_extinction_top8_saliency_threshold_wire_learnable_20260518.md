# Codex Routing Directive — TOP-8 Arbitrariness Extinction: Saliency Threshold 0.5 — Wire Existing Learnable

**Subagent**: `lane_arbitrariness_extinction_meta_lens_systematic_audit_20260518`
**Value ID**: `saliency_threshold_0.5_default_multiple_callsites`
**Resolution path**: `learned`
**Predicted ΔS**: [-0.003, -0.0005]
**Cost envelope**: $0 (pure engineering)
**Rank score per dollar**: 3.0

## Bug class

Saliency threshold 0.5 hardcoded across 3 callsites:
- `src/tac/saliency_inversion.py:174 threshold_quantile=0.5`
- `src/tac/codec_pipeline_sensitivity.py:519 high_threshold=0.5`
- `src/tac/learnable_saliency_threshold.py:100 init_threshold=0.5`

**The bug class is META**: `src/tac/learnable_saliency_threshold.py` ALREADY EXISTS as a learnable parameter wrapper (per Lane SI-V1 anchor). But it's NOT WIRED into the codec pipeline or saliency_inversion callsites. The LEARNED resolution path exists; the WIRING is missing.

## 5-path analysis

1. **experimental** — sweep threshold. Slow.
2. **analytical_solve** — ROC threshold optimization. Reasonable.
3. **formula** — N/A.
4. **learned** [RECOMMENDED] — `tac.learnable_saliency_threshold.LearnableSaliencyThreshold` already exists. WIRE IT.
5. **self_alien_tech** — N/A.

## Concrete next step ($0)

Pure engineering fix:

```python
# src/tac/codec_pipeline_sensitivity.py
from tac.learnable_saliency_threshold import LearnableSaliencyThreshold

@dataclass
class SensitivityPipelineConfig:
    learnable_threshold: bool = True   # NEW
    init_high_threshold: float = 0.5
    init_low_threshold: float = 0.1

# In pipeline forward:
if config.learnable_threshold:
    threshold = self.learnable_threshold(salience_scores)
else:
    threshold = config.init_high_threshold
```

Sister change in `saliency_inversion.py`. Add tests confirming the learnable threshold parameter receives gradient.

## Coupling

- META rule per CLAUDE.md "Subagent coherence-by-default": the LEARNED path exists but isn't wired = ORPHAN SIGNAL. This is the canonical fix.

## Exit criteria

1. `codec_pipeline_sensitivity.py` accepts `learnable_threshold=True` config
2. `saliency_inversion.py` likewise
3. Smoke confirms learnable threshold receives non-zero gradient
4. Empirical anchor confirms predicted ΔS lower bound
