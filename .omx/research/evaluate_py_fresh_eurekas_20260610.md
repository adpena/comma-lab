# evaluate.py fresh-eyes eurekas (operator-directed re-read, 2026-06-10) — AFSR-1 inputs

Line-cited from upstream/evaluate.py (113 lines). `[derivation from frozen evaluator source]`.

## E1 — THE RATE LAW COUNTS archive.zip ONLY; the inflate PROGRAM is rate-free (line 63)
`compressed_size = (submission_dir/'archive.zip').stat().st_size` — inflate.py/inflate.sh bytes are
NOT in the rate term. The interpreter is free real estate; only payload counts. Precedent: PR110's
mode catalog + Huffman codebook live in CODE (accepted by maintainers); PR101's inflate.py is ~480
LOC. ⇒ (a) the V6 inflate-as-interpreter thesis gets a literal subsidy — program logic, constants,
tables, procedural generators cost ZERO rate; (b) candidate action: migrate small payload-class
sections into code where compliance-defensible (one-video overfit is authorized; review norms bound
abuse, not the formula). AFSR-1's export should weigh data-vs-code placement per section. A
compliance-bounded audit of "largest defensible inflate.py" is a named follow-up.

## E2 — THE GT IS DEVICE-DEPENDENT (lines 39-42, 58)
CUDA GT = DaliVideoDataset (NVDEC decode); CPU GT = AVVideoDataset (PyAV). The ground truth PIXELS
differ per axis. ⇒ our flip map / atlas / cone were built vs PyAV GT = contest-CPU-correct; the
CUDA axis has its OWN (slightly different) flip map. AFSR-1 aims at CPU GT for the CPU axis —
correct — but the dual-axis submission gate needs CUDA-GT-aware verification, and the running
CUDA eval's transfer question is partly "how much does the GT decode difference move d_seg/d_pose."
This is the mechanism behind the known CPU/CUDA drift — now source-located.

## E3 — BOTH distortion terms are POOLED MEANS before the nonlinearity (lines 81-92)
`posenet_dists += sum; mean = total/600; score = 100*seg_mean + sqrt(10*pose_mean) + 25*rate`.
The sqrt applies to the MEAN pose over all 600 pairs ⇒ per-pair pose contributions are FUNGIBLE
inside the sqrt (a pose regression on pair A is exactly offset by equal improvement on pair B —
in d-domain, not score-domain). Confirms the composition algebra (compose in distortion domain)
from the evaluator source itself, and licenses CROSS-PAIR pose trading as a first-class move:
AFSR-1's curriculum may sacrifice pose on saturated pairs to buy it on cheap pairs at exactly 1:1.

## E4 — the denominator is computed LIVE (line 64): `sum(rglob('*'))` over the videos dir — equals
37,545,489 for the public set but is not a constant in code; keep reading it from the artifact.

## E5 — fixed seed 1234 + zip'd loaders (71-74): pairing relies on deterministic iteration order;
the first two tuple elements are discarded. Shape assert (77): inflated frames must be EXACTLY
(seq_len=2, H, W, 3) camera-size — the receiver contract.

Consumers: AFSR-1 (E1 export placement, E2 axis-aware aiming, E3 cross-pair curriculum) ·
the CUDA eval in flight (E2 interpretation) · V6 (E1 is the witness-program subsidy made literal).
