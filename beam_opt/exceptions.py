"""
beam_opt.exceptions
===================
Exception types raised by the framework.
"""

from __future__ import annotations


class SolutionNotConvergedError(RuntimeError):
    """
    The FEM solution did not reach the end of the load step.

    Raised instead of returning partial results, so that a non-converged
    solution never reaches the optimizer as a valid design evaluation.
    """
