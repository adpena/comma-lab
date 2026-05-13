# PR97 Public-Clone Quarantine - 2026-05-13

Context: developer preflight failed because the PR97 public intake clone was not
pristine. The clone had one untracked local binary:

- original path:
  `experiments/results/public_pr_intake_full/public_pr97_intake_20260505_auto/source/submissions/vibe_coder_final_boss/range_mask_codec.bin`
- file type: Mach-O 64-bit executable arm64
- bytes: `126984`
- sha256:
  `d9b27dfbf214d1c39825f2d0bf5d729ecf9f377169b452279c609578f18b8283`
- preserved copy:
  `experiments/results/pr97_dirty_clone_quarantine_20260513_codex/range_mask_codec.bin`

Disposition: the binary was copied to the ignored quarantine artifact path above
before removing the untracked copy from the public PR clone. This preserves the
local signal while restoring the clone to byte-pristine forensic state. No score
claim, archive claim, or public-PR source mutation is made from this file.

