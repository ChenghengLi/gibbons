"""
3D N² Queens MCMC Solver Package - Refactored

This package provides a clean, modular implementation of MCMC solvers
for the 3D N² Queens problem with both full and reduced state spaces.

Modules:
    - interfaces: Abstract base classes for Board and Solver
    - utils: Common utilities, JIT functions, and energy computations
    - board: Board state implementations (Full and Reduced)
    - solver: MCMC solver implementations
    - config: Configuration management
    - visualize: Visualization functions for solutions and energy histories
"""

from .interfaces import BoardInterface, SolverInterface
from .board import FullBoardState, ReducedBoardState
from .solver import FullStateSolver, ReducedStateSolver, create_solver
from .config import Config
from .visualize import (
    visualize_solution,
    visualize_latin_square,
    plot_energy_history,
    plot_averaged_energy_history,
    save_results,
    save_multiple_results,
    save_run_results,
    save_competition_format,
    create_run_output_folder,
    count_endangered_queens,
)

__all__ = [
    'BoardInterface',
    'SolverInterface',
    'FullBoardState',
    'ReducedBoardState',
    'FullStateSolver',
    'ReducedStateSolver',
    'create_solver',
    'Config',
    'visualize_solution',
    'visualize_latin_square',
    'plot_energy_history',
    'plot_averaged_energy_history',
    'save_results',
    'save_multiple_results',
    'save_run_results',
    'save_competition_format',
    'create_run_output_folder',
    'count_endangered_queens',
]
