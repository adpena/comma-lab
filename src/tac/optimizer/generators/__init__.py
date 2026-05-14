# SPDX-License-Identifier: MIT
"""Concrete sweep candidate generators for tac.optimizer.sweep_plugin.

Importing this package registers every shipped generator with the global
plugin registry. New generators should be added as a sibling module here
and imported below.
"""

from tac.optimizer.generators import apogee_intn

__all__ = ["apogee_intn"]
