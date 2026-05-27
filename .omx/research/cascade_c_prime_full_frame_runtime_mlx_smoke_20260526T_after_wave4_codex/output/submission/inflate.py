#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Top-level inflate.py per contest 3-arg contract (archive_dir output_dir file_list)."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
# SUBMISSION_PYTHONPATH_SHIM_OK:vendored_substrate_package_at_src_tac_substrates_cascade_c_prime_frame_1_segnet_waterfill_with_canonical_init_py_stubs_per_catalog_295_self_containment
sys.path.insert(0, str(HERE / "src"))
from tac.substrates.cascade_c_prime_frame_1_segnet_waterfill.inflate import main_cli as main
if __name__ == "__main__":
    sys.exit(main())
