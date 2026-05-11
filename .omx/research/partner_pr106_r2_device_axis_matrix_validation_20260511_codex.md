# Device-axis auth-eval matrix analysis

generated_at_utc: `2026-05-11T17:50:41Z`
score_claim: `false`
promotion_eligible: `false`

| Label | Axis | Scorer | Inflate | Score | Pose | Seg | Runtime SHA | Raw SHA |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| r1_cuda | contest_cuda | cuda | auto | 0.20739428085403283 | 3.281e-05 | 0.00064893 | 55989d263d4e | 235752e9c7e4 |
| r1_cpu | contest_cpu | cpu | auto | 0.2286802845175232 | 0.00016424 | 0.00063766 | 7a49e777d9d8 | 936d9c568d7a |
| r2_cuda | contest_cuda | cuda | auto | 0.20664588545741508 | 3.236e-05 | 0.0006426 | 7a49e777d9d8 | 5f65c70f59c7 |
| r2_cpu | contest_cpu | cpu | auto | 0.22809238271134513 | 0.00016402 | 0.00063196 | 9181713cd849 | 08675dc4d129 |

## Deltas vs baseline

- r1_cuda: score_delta=0.0007483953966177515, pose_delta=4.499999999999974e-07, seg_delta=6.330000000000029e-06, same_runtime=False, same_raw=False
- r1_cpu: score_delta=0.022034399060108123, pose_delta=0.00013188000000000002, seg_delta=-4.940000000000044e-06, same_runtime=True, same_raw=False
- r2_cpu: score_delta=0.021446497253930052, pose_delta=0.00013166, seg_delta=-1.064000000000002e-05, same_runtime=False, same_raw=False

## Notes

- This analysis is diagnostic only; use canonical auth-eval artifacts for score claims.
- A non-auto inflate_device_policy is never promotion-eligible.
- Raw-output SHA equality localizes drift after inflate; raw-output SHA mismatch localizes drift at or before inflate.
- Do not infer CPU or CUDA is globally better from this matrix; keep the conclusion per archive/runtime.
