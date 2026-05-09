# A5 Trust-Region q6-low45 macOS CPU Advisory - 2026-05-09

## Verdict

Measured advisory result: **0.21129939214393487** on macOS CPU.

This improves the prior q6-low50 advisory (`0.21336553243503031`) but is still
not competitive with the PR107/PR101 CPU anchors. It is a scoped A5
trust-region improvement, not a score claim and not promotion-eligible.

## Candidate

- Technique: `a5_score_marginal_trust_region_q6_low45`
- Candidate id: `a5_trust_q6_low0p45_20260509_codex`
- Archive: `experiments/results/pr101_frame_conditional_runtime_packet_trust_q6_low0p45_20260509_codex/packet/archive.zip`
- Archive bytes: `178138`
- Archive SHA-256: `dd725682e8dca781c908f826c117395c39cdffeba4ec2525f7799839eda3bfde`
- Runtime-tree SHA-256: `8c34b7442a60aec5c4196a4697d4da555095deae3ceb4057efb2f3981db2c140`
- q-bit schedule: `q6` for the lowest `270 / 600` score-marginal pairs, `q8` otherwise
- q-bit mean: `7.1`
- q-bit side-info SHA-256: `dbb44b4fae3a915ede1c2d6983758a8a2b6c99cb4a9cf09eb682708496d63a0f`

## Ladder Boundary

| low fraction | low pair count | archive bytes | SHA-256 prefix | status |
|---:|---:|---:|---|---|
| `0.35` | `210` | `178558` | `a02044880df561b4` | byte gate fail |
| `0.40` | `240` | `178348` | `35dcf5938d468416` | byte gate fail |
| `0.4467` | `268` | `178152` | `92d960b65baf57e7` | byte gate fail |
| `0.4484` | `269` | `178145` | `b3cacc23ca937f45` | byte gate fail by 1 B |
| `0.45` | `270` | `178138` | `dd725682e8dca781` | first byte-gate pass |
| `0.47` | `282` | `178054` | `48128c28183b75ab` | byte-gate pass, more distortion risk |

This pins the smallest tested q6 trust region that beats the PR101 brotli byte
anchor: **270 low-marginal pairs**.

## Advisory Eval

Command:

```bash
PYTHON=.venv/bin/python .venv/bin/python -u experiments/contest_auth_eval.py \
  --archive experiments/results/pr101_frame_conditional_runtime_packet_trust_q6_low0p45_20260509_codex/packet/archive.zip \
  --inflate-sh experiments/results/pr101_frame_conditional_runtime_packet_trust_q6_low0p45_20260509_codex/packet/inflate.sh \
  --upstream-dir upstream \
  --device cpu \
  --work-dir experiments/results/pr101_frame_conditional_runtime_packet_trust_q6_low0p45_20260509_codex/macos_cpu_advisory_work \
  --json-out experiments/results/pr101_frame_conditional_runtime_packet_trust_q6_low0p45_20260509_codex/contest_auth_eval.macos_cpu_advisory.json \
  --inflate-timeout 1800 \
  --evaluate-timeout 5400 \
  --keep-work-dir
```

Result:

- Canonical score: `0.21129939214393487`
- PoseNet distortion: `0.00004048`
- SegNet distortion: `0.00072565`
- Rate term: `0.11861475`
- Samples: `600`
- Inflate elapsed: `35.4 s`
- Evaluate elapsed: `410.9 s`
- Hardware: Apple Silicon macOS CPU advisory, not contest-CPU
- Review packet: `.omx/research/artifacts/a5_trust_q6_low45_result_review_20260509_codex.json`

## Interpretation

q6-low45 is a real local improvement over q6-low50:

- q6-low50: `0.21336553243503031`, `177928 B`, SegNet `0.00074546`
- q6-low45: `0.21129939214393487`, `178138 B`, SegNet `0.00072565`

The improvement comes from reducing the q6 region, which recovers SegNet more
than the extra `210 B` costs in rate. The result is still not competitive:
SegNet remains too high relative to the medal-band CPU anchors. Further scalar
low-fraction sweeps are likely low-EV unless paired with SegNet-boundary or
score-component marginals.

## Reactivation Criteria

- Replace scalar score-marginal selection with SegNet-boundary-aware or
  per-component score-marginal allocation.
- Keep the byte gate explicit: candidates must beat PR101 brotli bytes or have
  a separate exact reason to pay the rate term.
- Run exact contest-CUDA and contest-CPU only after advisory score is
  competitive with the PR101/PR107 anchor band and a lane claim is active.
