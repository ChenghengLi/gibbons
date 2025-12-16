"""
Abstract interfaces for Board and Solver classes.

These interfaces define the contract that all implementations must follow,
enabling polymorphism and clean separation of concerns.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
import jax.numpy as jnp


class BoardInterface(ABC):
    """
    Abstract interface for board state representations.
    
    A board state represents a configuration of N² queens on an N×N×N cube.
    Different implementations may use different internal representations
    (full state space vs reduced state space).
    
    Attributes:
        N: Board dimension (N×N×N cube with N² queens)
        queen_count: Number of queens (N²)
        energy: Current energy (number of attacking pairs)
    """
    
    @property
    @abstractmethod
    def N(self) -> int:
        """Board dimension."""
        pass
    
    @property
    @abstractmethod
    def queen_count(self) -> int:
        """Number of queens (N²)."""
        pass
    
    @property
    @abstractmethod
    def energy(self) -> float:
        """Current energy (attacking pairs count)."""
        pass
    
    @abstractmethod
    def get_queens(self) -> jnp.ndarray:
        """
        Get queen positions as N² × 3 array.
        
        Returns:
            Array of shape (N², 3) with (i, j, k) coordinates for each queen.
        """
        pass
    
    @abstractmethod
    def get_line_counts(self) -> Dict[str, jnp.ndarray]:
        """
        Get line count structures for O(1) energy updates.
        
        Returns:
            Dictionary mapping line family names to count arrays.
        """
        pass
    
    @abstractmethod
    def compute_energy(self) -> float:
        """
        Compute energy from scratch (ground truth).
        
        Returns:
            Number of attacking queen pairs.
        """
        pass
    
    @abstractmethod
    def copy(self) -> 'BoardInterface':
        """
        Create a deep copy of this board state.
        
        Returns:
            New BoardInterface instance with same state.
        """
        pass


class SolverInterface(ABC):
    """
    Abstract interface for MCMC solvers.
    
    A solver runs Metropolis-Hastings MCMC on a board state to find
    configurations with minimal energy (ideally zero).
    
    The solver should support:
    - Simulated annealing with various cooling schedules
    - Different energy treatments (linear, quadratic, log)
    - O(1) hash-based or O(N²) iterative energy computation
    """
    
    @abstractmethod
    def run(
        self,
        num_steps: int,
        initial_beta: float,
        final_beta: float,
        cooling: str,
        simulated_annealing: bool,
        energy_treatment: str = 'linear',
        energy_reground_interval: int = 0,
        verbose: bool = True
    ) -> Tuple[Any, jnp.ndarray, float]:
        """
        Run MCMC optimization.
        
        Args:
            num_steps: Number of MCMC steps to run
            initial_beta: Starting inverse temperature
            final_beta: Final inverse temperature  
            cooling: Cooling schedule ('linear', 'geometric', 'adaptive')
            simulated_annealing: Whether to use annealing (vs constant beta)
            energy_treatment: Energy function ('linear', 'quadratic', 'log', 'log_quadratic')
            energy_reground_interval: Steps between energy recalculations (0 to disable)
            verbose: Whether to print progress
            
        Returns:
            Tuple of (final_board_state, energy_history, acceptance_rate)
        """
        pass
    
    @abstractmethod
    def get_board(self) -> BoardInterface:
        """
        Get the current board state.
        
        Returns:
            Current BoardInterface instance.
        """
        pass
    
    @abstractmethod
    def set_seed(self, seed: int) -> None:
        """
        Set the random seed for reproducibility.
        
        Args:
            seed: Random seed value.
        """
        pass


class MCMCStepInterface(ABC):
    """
    Abstract interface for a single MCMC step.
    
    This separates the step logic from the solver loop,
    allowing different step implementations (hash-based, iterative).
    """
    
    @staticmethod
    @abstractmethod
    def step(
        state: Tuple,
        N: int,
        key: jnp.ndarray,
        beta: float,
        energy_treatment: Optional[callable] = None
    ) -> Tuple[Tuple, float, bool, jnp.ndarray]:
        """
        Execute a single MCMC step.
        
        Args:
            state: Current state tuple (implementation-specific)
            N: Board dimension
            key: JAX random key
            beta: Inverse temperature
            energy_treatment: Optional energy transformation function
            
        Returns:
            Tuple of (new_state, delta_energy, accepted, new_key)
        """
        pass
