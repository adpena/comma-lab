# Ledger 01 — L3 Harris tactical-communications lineage (2026-05-13)

**Lane:** `lane_expert_team_hardware_physics_future_alien_tech_20260513` (companion to master memo `expert_team_hardware_physics_future_alien_tech_20260513.md`).
**Persona:** L3 Harris Tactical Comms (Rochester NY / Lynchburg VA / Melbourne FL) — Falcon III handhelds, Falcon4 SDR, AN/PRC-117G/G manpacks, AN/PRC-152A handhelds. We design for ≤ 2 W TX power, sub-1-kbps voice/data, CCSDS-style FEC over fading HF/VHF/UHF channels, and Hexagon/ARM-Cortex DSPs.
**Mode:** READ-ONLY engineering-analog derivation. `research_only=true`. `score_claim=false`. `promotion_eligible=false`. `ready_for_exact_eval_dispatch=false`. NO archive bytes mutated.
**Evidence:** `[engineering-analog]`, `[mathematical-derivation]`, `[literature-prediction]`.

---

## 0. The L3 Harris frame

We do not compress to a CPU's preferences. We compress to the **channel's capacity**. The receiver is **known**, the channel has **known impairments**, and the encoder is **co-designed** with the receiver. This is exactly the contest setup: SegNet + PoseNet are the known receivers; the inflate runtime is the known channel; the archive is the encoder output. Everything below is a port of a known-working tactical-comms technique.

Reference standards we operationalize daily:
- **MIL-STD-188-110D**: HF data modems (3 kHz channels, OFDM/QPSK).
- **MIL-STD-3005**: MELP-2400 voice coder (2.4 kbps, NSA-blessed).
- **MIL-STD-188-181C**: UHF SATCOM.
- **CCSDS 131.0-B-3**: TM Synchronization and Channel Coding (Reed-Solomon, convolutional, turbo, LDPC).
- **CCSDS 231.0-B-3**: TC Synchronization and Channel Coding.

---

## 1. Falcon III MELP-on-pose-residuals

### 1.1 Background

MELP (Mixed-Excitation Linear Prediction, McCree & Barnwell 1995) models speech as a 10-pole LPC filter excited by a mixed pulse/noise source. MELP-2400 fits speech in 2.4 kbps:

- 10 LPC line spectral frequencies (LSF), 25 bits/frame
- pitch, 7 bits/frame
- voicing strength per band, 4 bits/frame
- gain, 5 bits/frame
- aperiodic flag, 1 bit/frame
- Fourier magnitudes, 8 bits/frame
- Total 54 bits/22.5 ms frame = 2.4 kbps

The breakthrough: **decorrelate via LPC analysis filter, then code only the residual**. Most of the bits are spent on the filter coefficients (which are slowly-varying); residual is sparse.

### 1.2 Contest analog

Pose trajectory through the 1200-frame video is a **6-DOF time series at 20 FPS**. Vehicle dynamics give it a power spectrum concentrated below ~5 Hz (yaw rate, pitch rate, roll rate are bandlimited by tire dynamics + suspension). LPC fits this signal class.

```python
# Inflate-time decoder (NumPy pseudocode, ~30 LOC)
def lpc_decode_pose_trajectory(lpc_coeffs, residuals, initial_pose):
    """
    lpc_coeffs: shape (6, p) — p-pole LPC for each pose axis
    residuals: shape (1199, 6) — single-byte innovations per axis per pair
    initial_pose: shape (6,) — first pose, 24 bytes
    Returns: pose_trajectory shape (1200, 6)
    """
    p = lpc_coeffs.shape[1]
    poses = np.zeros((1200, 6))
    poses[0] = initial_pose
    for t in range(1, 1200):
        for axis in range(6):
            # AR(p) prediction
            pred = sum(lpc_coeffs[axis, k] * poses[t-1-k, axis] for k in range(min(p, t)))
            poses[t, axis] = pred + dequantize(residuals[t-1, axis])
    return poses
```

### 1.3 Bit budget

- LPC coeffs: 6 axes × 4 poles × 8 bits = 24 bytes
- Initial pose: 6 floats × 4 bytes = 24 bytes
- Residual quantizer scales: 6 × 4 bytes = 24 bytes
- Residuals: 1199 pairs × 6 axes × 8 bits = ~7.2 KB

**Total: ~7.3 KB vs PR101's ~24 KB pose stream.** Savings: ~17 KB.

### 1.4 Score-impact prediction

Rate term: -17 KB / 37.5 MB = -0.00045 to -0.00075 (per CLAUDE.md PR106 marginal pose sensitivity 2.71×).

Distortion term: 8-bit residual quantization at 5 Hz signal bandwidth → SNQR ≈ 48 dB → pose error contribution ≤ 1e-5 (well below current PR106 r2 pose_avg = 3.4e-5). Net pose distortion change: marginal.

**Net score: -0.0006 to -0.0010** [mathematical-derivation, literature-prediction].

### 1.5 Reactivation / coordination

Cross-link with sister Bell-Labs memo H3 (linear-prediction-coded pose) — likely converges on same primitive. Coordinate via byte-stream-shape primitive registry. If empirical falsification surfaces, treat as DEFER-pending-quantizer-redesign (e.g. switch to vector quantizer or arithmetic-coded innovations) — NEVER KILL per CLAUDE.md non-negotiable.

---

## 2. Reed-Solomon entropy-multiplexed ZIP headers

### 2.1 Background

L3 Harris radios run RS(255, 223) over deep-fading channels. Outer code: Reed-Solomon over GF(256), corrects up to 16 erasures per 255-byte codeword. Inner code: convolutional or turbo.

**Insight:** the RS redundancy bytes are 32 elements of GF(256) determined by the 223 data bytes via polynomial evaluation. If we **choose the data bytes** to make the redundancy bytes equal *useful content*, both serve double duty.

### 2.2 Contest analog

A contest archive's ZIP framing carries:
- ZIP local file header: ~30 bytes
- ZIP central directory header: ~46 bytes per file
- ZIP end-of-central-directory record: ~22 bytes
- Brotli stream header: ~3-6 bytes
- Total: ~200-400 bytes of low-entropy structural overhead

Most fields are **constrained but not fully fixed** — file names, timestamps, comment fields, extra fields can be chosen by the encoder. The ZIP **extra-field** is an arbitrary byte vector that some parsers ignore but `unzip` preserves; up to 65 KB / file in principle.

### 2.3 Bit budget

Hide 200-400 bytes of pose-residual stream inside ZIP extra-fields and comment fields:
- 1 file × 200 bytes extra-field = 200 bytes free
- 1 file × 200 bytes comment = 200 bytes free
- Total recovered: ~400 bytes

### 2.4 Score-impact prediction

400 bytes / 37.5 MB = -0.00011 score.

### 2.5 Caveats and reactivation

- Some ZIP parsers reject non-canonical extra-fields. **Test against contest's exact `inflate.sh` runtime** before committing.
- Sister codex memo `pr106_r2_packetir_identity_zip_comment_20260513_codex.md` explores ZIP comment field; coordinate.
- If extra-field path is rejected, DEFER-pending-runtime-acceptance-research.

---

## 3. Hexagon DSP HVX-aligned inflate kernel

### 3.1 Background

comma.ai's Snapdragon 845 has Hexagon 685 with HVX (Hexagon Vector Extensions): 128-byte SIMD vector, VLIW, ~5 GHz clock. Mobile-deploy of neural inference uses HVX intrinsics to hit 50+ TOPS at ~1 W TDP.

### 3.2 Contest analog

The contest scoring is done on T4 CUDA (8 TFLOPS FP32, 70 W TDP) or CPU x86_64 (multi-core, ~50 W). Neither is HVX. But if our `inflate.py` decoder kernels were **structured as if HVX were the target** — 128-byte aligned tensors, no scatter/gather, fixed-stride memory access, no FP exceptions — they would (a) run faster on T4 (~2-4× inflate wall-clock reduction) due to coalesced memory access, and (b) be byte-deterministic across CPU/CUDA.

### 3.3 Implementation sketch

Use PyTorch `tensor.contiguous()` after every op that might break alignment. Avoid `torch.gather`, prefer `torch.index_select` with contiguous index tensors. Use `torch.nn.functional.unfold` for windowed access.

### 3.4 Score-impact prediction

**No direct score gain.** Indirect: 2-4× faster inflate frees compute for more elaborate decoder logic within the 30-min budget. Enables candidates that currently bump the wall-clock ceiling.

---

## 4. Bit-truncated Costas-loop pose tracking

### 4.1 Background

Costas loop (Costas 1956, *Proc. IRE*) is the canonical phase-locked-loop for suppressed-carrier demodulation. A 2-pole IIR with phase-error feedback tracks a continuously-varying phase trajectory at < 1 dB excess over Cramér-Rao bound.

### 4.2 Contest analog

Yaw, pitch, roll over 1200 frames is a slowly-varying 3D trajectory. Costas-style tracking represents it as:
- Initial state: 6 floats × 4 bytes = 24 bytes
- Loop filter coefficients: 2 poles × 6 axes × 4 bytes = 48 bytes
- Per-frame phase-error innovations: 1199 × 6 × 1 byte = 7.2 KB

**Total: ~7.3 KB** — same byte budget as §1 MELP-on-pose-residuals; alternative formulation.

### 4.3 Score-impact prediction

Similar to §1: -0.0006 to -0.0010 [mathematical-derivation].

### 4.4 Which to pick (MELP vs Costas)

MELP is **predictive**: fit LPC, code residual. Costas is **tracking**: phase-error feedback. Both have similar byte budgets.

**Recommend MELP** because it has a well-defined empirical bound (SNQR scales with quantizer bits) and the decoder is 30 LOC. Costas decoder requires careful pole placement for stability; harder to verify.

---

## 5. Status / cross-references / next steps

- **Companion:** master memo `expert_team_hardware_physics_future_alien_tech_20260513.md`.
- **Sister subagents:**
  - Bell Labs lineage (`expert_team_signal_processing_bell_labs_20260513.md`) — Technique H3 (LPC pose) overlaps with §1 here. **Coordinate**: pick one canonical implementation, declare the other as dual derivation.
  - Lincoln Lab lineage (`expert_team_signal_processing_lincoln_lab_20260513.md`) — likely covers radar-style pulse compression.
  - NSA SIGINT (`expert_team_signal_processing_nsa_sigint_20260513.md`) — likely covers blind-signal separation, may overlap §1.
- **Active codex work:** `pr106_r2_packetir_identity_zip_comment_20260513_codex.md` overlaps §2.
- **Wire-in hooks:** declared in master memo §9.
- **Reactivation:** all techniques in this ledger are `research_only`. None proceed to archive-bytes-change without (a) the score-aware-loss-from-byte-zero discipline (CLAUDE.md HNeRV parity lesson 1), (b) explicit grand-council review per CLAUDE.md "Design decisions" non-negotiable, (c) Catalog #123 score-gradient saliency discipline for any QAT-adjacent work.

**Per CLAUDE.md "KILL is LAST RESORT" non-negotiable:** every technique above is DEFER-pending-research, never KILLED, with explicit reactivation criteria.
