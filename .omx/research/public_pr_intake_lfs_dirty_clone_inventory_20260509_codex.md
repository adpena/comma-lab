# Public PR Intake LFS Dirty Clone Inventory - 2026-05-09

<!-- generated_at: 2026-05-09T15:00:00Z -->
<!-- evidence_grade: custody_repair; no score claim -->

## Scope

`tac.preflight` blocked on `check_public_pr_intake_clones_pristine`: two
detached public PR source clones had binary/LFS payload files materialized in
the working tree while their upstream Git state stores LFS pointer files.

To preserve signal before restoring clone pristine state, the materialized file
sizes and SHA-256 hashes were recorded below. This ledger does not promote or
alter any score claim.

## Clone Heads

- PR90 clone:
  `experiments/results/public_pr_intake_full/public_pr90_intake_20260505_auto/source`
  at `cce857392701e73861ad513d34906faba523f719`.
- PR104 clone:
  `experiments/results/public_pr_intake_full/public_pr104_intake_20260505_auto/source`
  at `f1c59d895325f2d2835e843ce72be3443983a4b4`.

## Materialized Payload Inventory

| clone | bytes | sha256 | path |
|---|---:|---|---|
| pr90 | 4007665 | `9f17e9dc0c511bff9d5265491a33e08ec99e611457837fcd009e82f524afd771` | `submissions/qrepro/assets/animated_triptych.gif` |
| pr90 | 64806 | `f73cca773b3470403bc2ee5a5541f9fca606650d4d522e8a0dc3d66703adac67` | `submissions/qrepro/assets/archive_components.png` |
| pr90 | 4130149 | `6c7e7af36d3c8026cd3916a8db24bb9e05146eb96a38fdaad2e118cdf9ae1358` | `submissions/qrepro/assets/best_worst_gallery.png` |
| pr90 | 118569 | `e0bd60535e4080f0f5f4b60eb538b7731cdec7cf82d9378119926f0c80a1dbc7` | `submissions/qrepro/assets/metric_sensitivity.png` |
| pr90 | 658662 | `51f8efd0ca8b33da9c58042a5fc9adf3de02dc1cb825cda77d75f15907bdc00f` | `submissions/qrepro/assets/per_frame_score_timeline.png` |
| pr90 | 37117 | `1a8c2cf68beac7a8371c6787e9b5bea6357467a6199f4ed34f4645da687fa958` | `submissions/qrepro/assets/qrgb_basis_gallery.png` |
| pr90 | 68769 | `836e862f95bc096b0a2ab6aa17ea63684ddcbe590a3c1ea464098d2a04de6d8f` | `submissions/qrepro/assets/qrgb_correction_maps.png` |
| pr90 | 92165 | `8cd34dafabb378601cb58a6c07903654b1b9190b790ba71384f197205c4ef2cb` | `submissions/qrepro/assets/qrgb_residual_heatmap.png` |
| pr90 | 2593794 | `84ccc8fc6b85fcbea64e4b480eed07c8df50f3c6f9850eac75564896354122e9` | `submissions/qrepro/assets/reconstruction_triptych.png` |
| pr90 | 84303 | `f68467acd5548f2608c3bafe43743b696aa49687a338c3b060415143b25aa214` | `submissions/qrepro/assets/score_breakdown.png` |
| pr90 | 74126 | `c0a79fca8c7f574e1298f03063d3cb1bf04ff15b957c071e269c8b804b00c6a0` | `submissions/qrepro/assets/semantic_boundary_profiles.png` |
| pr90 | 101413 | `490d7388980db41183282d04d72bcd32bdc8a4af1638fdb1d65e46163e42b8c8` | `submissions/qrepro/assets/semantic_class_timeline.png` |
| pr90 | 69160 | `272eb9651c67ff52740895b83f230ba6e521c11c1720ceb17a68c51072e5d0c1` | `submissions/qrepro/assets/semantic_codec_components.png` |
| pr90 | 53976 | `e412ca96232a8c6306ab6e28ee66d7d9a0f60bb81e92deb4ef21ca7bb6ee5f42` | `submissions/qrepro/assets/semantic_codec_flow.png` |
| pr90 | 131101 | `589661d97c90588445f3a7008a5d5cb878b560610b04ee60b26611bd667499f1` | `submissions/qrepro/assets/semantic_decomposition.png` |
| pr90 | 245334 | `bdceccea5d1d71519a299087c183626d4a5d50dcd4d61f5707112328e058495f` | `submissions/qrepro/assets/stack_overview.png` |
| pr104 | 178637 | `6564c32a9edeeaf08abd7f0ea673ba2fda23444605ca207eb4ba794cc66797b8` | `submissions/qhnerv_ft_best/archive.zip` |

## Repair

After this inventory, restore the detached clones to pristine upstream state
with:

```bash
git -C experiments/results/public_pr_intake_full/public_pr90_intake_20260505_auto/source checkout -- .
git -C experiments/results/public_pr_intake_full/public_pr104_intake_20260505_auto/source checkout -- .
```

This restoration removes working-tree LFS materialization from source clones
only. The payload hashes above preserve the materialized signal for future
forensics if it is needed.
