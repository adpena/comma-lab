# SPDX-License-Identifier: MIT
"""Analysis artifacts and feature builders for scorer-aware compression.

This package is for measured or derived telemetry that describes the contest
video, scorers, public archives, or candidate outputs. It should not dispatch
GPU jobs or claim scores. Analysis modules emit typed JSON records consumed by
``tac.optimization`` or archive builders.
"""
