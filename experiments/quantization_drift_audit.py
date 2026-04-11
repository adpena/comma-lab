#!/usr/bin/env python3
"""INT8 dequantization drift audit — thin wrapper around tac.quantization_audit."""
from tac.quantization_audit import main

if __name__ == "__main__":
    raise SystemExit(main())
