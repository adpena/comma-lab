"""C6 runtime package (inflate-time only — no scorer imports)."""
from tac.substrates.c6_e4_mdl_ibps.architecture import (
    EVAL_HW,
    MDLIBPSConfig,
    MDLIBPSSubstrate,
    NUM_PAIRS,
)
from tac.substrates.c6_e4_mdl_ibps.archive import (
    IBPS1_MAGIC,
    MDLIBPSArchive,
    pack_archive,
    parse_archive,
)
from tac.substrates.c6_e4_mdl_ibps.ib_decoder import IBDecoder
from tac.substrates.c6_e4_mdl_ibps.ib_encoder import IBEncoder
from tac.substrates.c6_e4_mdl_ibps.inflate import inflate_one_video, main_cli
__all__ = ["inflate_one_video", "main_cli", "MDLIBPSConfig", "MDLIBPSSubstrate",
           "pack_archive", "parse_archive", "IBPS1_MAGIC"]
