# 2026-04-04 ROI two-pass prototype summary

## goal
Test a tiny fixed-ROI two-pass proxy for a segmentation-guided codec against the promoted 3.54 floor.

## result

- baseline current_workflow score: `3.54`
- prototype current_workflow score: `5.73`
- baseline archive bytes: `1,901,606`
- prototype archive bytes: `1,472,589`
- baseline local rule_faithful estimate: `3.546277389901901`
- prototype local rule_faithful estimate: `5.736802653194266`

## decision

Reject. The byte savings were real, but the semantic/task distortion increase was far too large.

## interpretation

A naive fixed ROI is not enough. If a segmentation-guided two-pass architecture is going to work, it likely needs:
- a much better mask
- smoother boundaries
- better temporal consistency
- more careful byte budgeting for the protected region
