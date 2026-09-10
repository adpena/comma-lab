"""Native audit-bundle entry point for the real retained RLC1 parity test."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.ddm_rlc1_reference_controls import main

if __name__ == '__main__':
    main()
