# ddm_t1h — ZERO-BYTE pose-coefficient re-solve headroom on the SHIPPED dxi (PR133's mechanism, our 97% axis)

**Fired by MAIN on hx1's NEXT_IF_RESUMED #1 (memo ddm_hx1_pr_wave_harvest_20260817.md,
6494d8065e).** PR 133's eval-bot-confirmed move (0.172141 → 0.165780 [contest-CUDA T4]) was
**89.5% a byte-frozen coefficient re-solve**: greedy coordinate search over ALREADY-TRANSMITTED
integer codes, exact PoseNet forward as the accept oracle, uint8 rounding INSIDE the loop, ZERO
added bytes. Our pose term (√(10·6.88e-6) ≈ 0.0083) is **97.2% of the remaining sub-0.15 gap**,
and nobody has measured how far our SHIPPED pose/dxi coefficients sit from their per-pair optimum.

**Honest prior, stated up front:** our d_pose 6.88e-6 is ~2 orders better than PR133's starting
point — headroom may be small. That is WHY this arm measures FIRST and only then solves. A
near-zero headroom verdict is valuable (it closes the axis at the coefficient level and hardens
the routing arithmetic); do not manufacture a win.

Standing laws: NO launches/Modal/paid (sealed fire-order to MAIN if a candidate emerges) · NO
n600 SegNet runs · the pose accept oracle is **exact CPU-torch PoseNet** (the pk4-r3 authority
precedent: torch-CPU parity 2.29e-5 vs retained custody; NEVER MPS, NEVER the advisory
instrument — me1 measured it 21× off on pose) · NO Metal/MLX allocation while the hg1 burn holds
the device (~23:20Z; CPU-torch only) · ALWAYS KEEP THE PAYLOAD (/Volumes/APDataStore/pact/ddm_t1h/)
· serializer commits w/ POST-EDIT sha · .py = 2 review passes · upstream/ READ-ONLY.

## Build order

1. **Recall + custody**: locate the shipped pose/dxi section in the rr4 archive (sha 35ac2b9b…,
   candidate_runtime custody) — the exact integer codes, their dequant law, and the decode path
   that consumes them (verify at the receiver source; never assume the format). Read hx1's PR133
   analysis note (APDataStore/ddm_hx1/notes/) for the mechanism's exact shape — search over
   TRANSMITTED codes, decode-side unchanged.
2. **THE HEADROOM MEASUREMENT (the arm's reason to exist, $0)**: per-pair, hold everything
   frozen except one pose coefficient at a time; evaluate d_pose through the EXACT decode path +
   CPU-torch PoseNet; measure the per-pair gap between shipped and per-coordinate-optimal
   d_pose. Aggregate: total achievable Δd_pose at ZERO added bytes → ΔS via √(10·d_pose).
   Deliver the headroom NUMBER with per-pair distribution before any solve.
3. **IF headroom ≥ 1e-4 S**: run the full greedy coordinate re-solve (PR133's loop shape:
   coordinate proposals · exact-oracle accept · uint8-in-loop · multiple passes to
   convergence · effort-matched honesty — report the pass curve). The output codes REPLACE the
   shipped section byte-for-byte-in-length (zero added bytes by construction — assert it).
   Byte-close → determinism repeat → sealed T4 fire-order (falsifiers: bytes UNCHANGED to the
   byte · d_seg UNCHANGED to all digits · d_pose ≤ measured local value within the CPU↔CUDA
   pose band from the #1054 receipts).
4. **IF headroom < 1e-4 S**: the coefficient level is CLOSED on this vehicle — write the
   verdict with the measured number + per-pair distribution; name what it implies for the pose
   axis (routes remaining pose value to the js8 joint/nonlinear line). verdict_scope: instance
   (this carrier's shipped coefficients on this archive).

## Prediction-trial framing (record in the memo)

This is the THIRD live discriminator alongside fx1's mixer and the QAT continuation: a t1h win
is a constrained-solve OPERATOR win (alive-subalgebra branch); a near-zero headroom supports
GS1-PRED (new-checkpoint). Report which prediction your outcome supports.

## Deliverables

1. The headroom number + per-pair distribution (the memo's headline, whatever it is).
2. If solving: the re-solved section + byte-close receipts + sealed fire-order for MAIN.
3. `.omx/research/ddm_t1h_pose_coeff_resolve_headroom_20260817.md` — mechanism receipts ·
   oracle-authority receipts (CPU-torch parity) · the pass curve · STORES CONSULTED.
4. Final message: headroom in S units + the fork taken + NEXT_IF_RESUMED. End with the
   own-vehicle frontier line + whether your unit moved it (it cannot — only MAIN's fire can).
