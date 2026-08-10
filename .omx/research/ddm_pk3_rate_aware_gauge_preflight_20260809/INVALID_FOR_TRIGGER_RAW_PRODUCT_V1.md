# Preserved PK3 v1 payload exclusion

The retained candidate bank under
`/Volumes/VertigoDataTier/pact/ddm_pk3_20260809/retained/candidates/` was produced
concurrently from the initial untracked runner before receiver semantics were
corrected. Its provenance within the shared workspace is unknown. The bytes and
checkpoints are preserved; none are deleted or overwritten.

**Disposition:** `INVALID_FOR_TRIGGER_RAW_PRODUCT_V1_PRESERVED_NOT_COUNTED`.

It cannot satisfy or fail the PK3 charter trigger because:

- its primary metric was low-resolution raw stored-factor `C @ B`, while the
  shipped receiver evaluates `C @ normalized_basis(B) / sqrt(12)` after bicubic
  upsample, mean removal, and row RMS normalization;
- its scale step multiplied the basis scale and reciprocally divided the
  coefficient scale, which changes the shipped receiver field because the basis
  magnitude is normalized away;
- its rotations and shears used an uncorrected raw inverse mixing law;
- its candidate identity did not bind the receiver metric or corrected transform
  law, so cached receipts cannot be safely upgraded in place.

The final valid run is isolated under
`/Volumes/VertigoDataTier/pact/ddm_pk3_20260809/receiver_v7/` with a warning-free
receiver metric, `normalization_aware_H_D_v2`, and the exact PK3 runner source
hash in every candidate identity. The preserved `receiver_v2/`, `receiver_v3/`,
`receiver_v4/`, `receiver_v5/`, and `receiver_v6/` passes are invalid,
superseded, or source-drift-aborted and do not enter the final denominator.
