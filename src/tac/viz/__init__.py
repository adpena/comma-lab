"""Visualization modules for TTO analysis and diagnostics.

This package provides reusable visualization functions designed to be
chained after authoritative evaluation in automated pipelines (Modal,
Kaggle, local).

Modules:
  - analysis_panels: 6-panel multipane analysis (GT vs reconstruction vs error)
"""

from __future__ import annotations

from tac.viz.analysis_panels import generate_analysis_panels

__all__ = ["generate_analysis_panels"]
