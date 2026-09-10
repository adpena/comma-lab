# ddm_gdc1 preregistered scorer-free falsifier

Date: 2026-09-10. Owner: `ddm_gdc1`. Axis: `[macOS-CPU scorer-free exact-field measurement, n600]`.

The tested construction is an ordered scanline partition program. For every image row it finds the
minimum-Hamming piecewise-constant representation with at most `K` runs, with endpoints restricted to
true target transitions. The program serializes row-start classes, transition counts, row-to-row delta
positions, and destination classes. The receiver reconstructs the complete field from only those counted
bytes. Generic dynamic programming, delta inversion, and rasterization are free receiver code; every
video-selected transition is counted.

The complete roster is `K = 4, 6, 8, 12, 16, 24`; all points are full n600. Selection minimizes
`packet_bytes + 0.2909 * mismatches`. All program packets, deterministic coder repeats, fitted renders,
receiver renders, and the winning real residual race must be retained.

Pre-run prediction: `K=8` is expected to be the knee. The source-only row census has 584,354 runs over
230,400 rows; 214,011 rows (92.8868%) already have at most eight runs. The remaining 16,389 rows contain
the topology tail. The predicted `K=8` mismatch band is 80,000–300,000 sites and the predicted packet
band is 60,000–100,000 B. The central estimate (`M=180,000`, `P=80,000`) gives 132,362 B, so the prior is
that this formulation will miss the current 94,010 B replaceable-tail gate, while the full curve is worth
measuring because the low end could pass.

PASS is physical `packet + best real coded residual <= 94,010 B`. The transferred GF1 screen is
`packet + 0.2909*mismatches < 94,010.265605... B`. A negative closes only this finite ordered-scanline
grammar and roster; it is not a family result for learned implicit generators, vector/SDF programs, or
compression-aware joint transition fitting.

