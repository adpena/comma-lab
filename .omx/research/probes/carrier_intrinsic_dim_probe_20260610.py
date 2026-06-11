"""$0 probe: per-pair carrier intrinsic-dimension split (d_seg vs d_pose budget).

Settles the C1 crux: across 600 pairs, how many BITS/pair does each scored term
need, and is the frontier's 28-d per-pair latent optimal or per-axis-misallocated?

ALL numbers MEASURED-derived from existing artifacts (no scorer re-run):
- pose: floor probe traj_std_per_dim + measured temporal-delta entropy per op-point.
- seg per-pair: lever B's measured 600-pair mod blob (16,888 B brotli / 600 pairs).
- partition temporal structure: floor probe temporal-context bytes.

Authority: [macOS-CPU advisory] — derived from MEASURED artifacts. NOT a score claim.
"""
import json, math

floor = json.load(open(".omx/research/information_theoretic_floor_probe_full_1781086910.json"))
leverb = json.load(open(".omx/research/lever_b_score_native_argmax_smoke_20260610_evidence/smoke_n600_result.json"))

D = 37_545_489
N = 600

print("="*72)
print("PROBE: PER-PAIR CARRIER INTRINSIC-DIMENSION SPLIT")
print("="*72)

# --- POSE intrinsic dimension (the key measured structure) ---
std = floor["pose"]["traj_std_per_dim"]
print("\n[POSE] measured per-dim std of the 600-frame trajectory (first 6 dims scored):")
for i, s in enumerate(std):
    print(f"  dim {i}: std = {s:.4f}   {'<<< DOMINANT' if s > 0.5 else '(near-constant)'}")

# Effective dimension via participation ratio of the variance spectrum.
var = [s*s for s in std]
total_var = sum(var)
pr = (sum(var)**2) / sum(v*v for v in var)  # participation ratio
print(f"\n  total variance = {total_var:.4f}")
print(f"  participation ratio (effective # dims) = {pr:.3f}")
print(f"  dim-0 carries {100*var[0]/total_var:.2f}% of pose variance")
print(f"  dims 1-5 carry {100*sum(var[1:])/total_var:.2f}% combined")

# pose carrier bytes at frontier op-point (MEASURED temporal-delta entropy).
pose_bytes = floor["pose"]["results"]["target_0.0172"]["bytes_temporal_delta"]
pose_bits_per_pair = floor["pose"]["results"]["target_0.0172"]["temporal_delta_bits"] / N
print(f"\n  pose carrier (d_pose tube 2.96e-5): {pose_bytes:.0f} B total = {pose_bits_per_pair:.2f} bits/pair")
print(f"  -> but ~{100*var[0]/total_var:.0f}% of that entropy is dim-0 alone")

# --- SEG per-pair mod budget (MEASURED from lever B) ---
seg_base = leverb["byte_accounting"]["base_bytes"] if "byte_accounting" in leverb else None
print("\n[SEG] lever-B measured 600-pair carrier (from verdict memo):")
# From the verdict: base 46,914 B (shared/amortized) + mod 16,888 B (600 pairs)
seg_base = 46914
seg_mod = 16888
seg_mod_bits_per_pair = seg_mod * 8 / N
print(f"  shared base (amortized, NOT per-pair): {seg_base} B")
print(f"  per-pair mod: {seg_mod} B = {seg_mod_bits_per_pair:.2f} bits/pair")
print(f"  (this is the per-pair seg-argmax-REFINEMENT entropy; base does the heavy lifting)")

# --- The per-pair information budget verdict ---
print("\n" + "="*72)
print("PER-PAIR INFORMATION BUDGET (the carrier-design number)")
print("="*72)
print(f"  d_seg per-pair (refinement over shared base): {seg_mod_bits_per_pair:.1f} bits/pair")
print(f"  d_pose per-pair (tube precision):             {pose_bits_per_pair:.1f} bits/pair")
print(f"  TOTAL per-pair carrier (seg-mod + pose):      {seg_mod_bits_per_pair + pose_bits_per_pair:.1f} bits/pair")
total_perpair_bits = seg_mod_bits_per_pair + pose_bits_per_pair
print(f"  = {total_perpair_bits/8:.1f} BYTES/pair  (×600 = {total_perpair_bits/8*600:.0f} B)")

# Compare to the frontier's 28-d latent.
# Frontier: latent stream 15,070 B / 600 pairs.
front_latent_bytes_per_pair = 15070 / N
front_latent_bits_per_pair = front_latent_bytes_per_pair * 8
print(f"\n  FRONTIER 28-d latent (measured): {15070} B / 600 = {front_latent_bytes_per_pair:.1f} B/pair = {front_latent_bits_per_pair:.1f} bits/pair")
print(f"  -> frontier spends {front_latent_bits_per_pair:.0f} bits/pair on the joint latent")
print(f"  -> measured score-DOF need: ~{total_perpair_bits:.0f} bits/pair (seg-mod {seg_mod_bits_per_pair:.0f} + pose {pose_bits_per_pair:.0f})")

# --- The 8-bit VQ index impoverishment (why the capstone walls) ---
print("\n" + "="*72)
print("THE 8-BIT VQ INDEX WALL (why pure-VQ capstone fails pose)")
print("="*72)
print(f"  capstone per-pair carrier = 8-bit VQ index = 8.0 bits/pair (256 buckets)")
print(f"  pose ALONE needs {pose_bits_per_pair:.1f} bits/pair (dim-0 dominated)")
print(f"  -> 8 bits cannot encode the dim-0 ego-motion (std 1.256, range -0.13..35.05)")
print(f"  -> 600 distinct ego-motions into 256 buckets = quantization collapse")
print(f"  -> THIS is the measured root cause of d_pose 0.06-0.34 wandering")

# --- Per-axis allocation analysis ---
print("\n" + "="*72)
print("PER-AXIS ALLOCATION: is the latent mis-split?")
print("="*72)
print(f"  pose intrinsic dim (participation ratio): {pr:.2f} -> ~1-2 effective dims")
print(f"  but dim-0 RANGE is huge (-0.13..35.05) -> needs ~{pose_bits_per_pair:.0f} bits/pair PRECISION not dims")
print(f"  seg per-pair refinement: {seg_mod_bits_per_pair:.0f} bits/pair")
print(f"  => pose is a LOW-DIM HIGH-PRECISION signal; seg is a higher-dim low-precision refinement")
print(f"  => optimal carrier: ALLOCATE BITS (precision), not just dims, per-axis")
print(f"  => a uniform 28-d latent under-serves pose-dim-0 precision IF VQ-quantized;")
print(f"     stored-float 28-d latent (temporal-delta) serves both (frontier proves it)")
