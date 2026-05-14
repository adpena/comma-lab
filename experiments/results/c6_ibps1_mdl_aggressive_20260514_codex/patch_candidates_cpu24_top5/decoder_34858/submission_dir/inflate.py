"""C6 MDL-IBPS inflate entry — delegates to vendored substrate."""
import sys

from tac.substrates.c6_e4_mdl_ibps.inflate import main_cli

if __name__ == '__main__':
    sys.exit(main_cli())
