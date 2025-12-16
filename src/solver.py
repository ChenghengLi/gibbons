"""
MCMC Solver implementations for 3D N² Queens Problem.

This module provides two solver implementations:
- FullStateSolver: For full state space (moves queen to any empty cell)
- ReducedStateSolver: For reduced state space (changes k-coordinate only)

Both support:
- hash: O(1) energy updates using line counting
- iter: O(N²) energy updates using iterative attack checking (counts attacking pairs)
- endangered: O(N²) energy updates counting endangered queens
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Dict, Tuple, Optional, Callable, Any
import time

from .interfaces import SolverInterface, BoardInterface
from .board import FullBoardState, ReducedBoardState
from .utils import (
    get_energy_treatment,
    compute_energy_from_line_counts,
    compute_delta_energy_treated,
    compute_energy_delta_and_update_lines,
    compute_reduced_energy_delta_and_update_lines,
    compute_beta_linear,
    compute_beta_geometric,
    compute_cooling_rate,
    initialize_line_counts_from_queens,
    check_attack_jit,
)


# =============================================================================
# O(N²) Iterative Attack Checking Functions
# =============================================================================

@jax.jit
def is_queen_endangered(queen_idx: jnp.ndarray, queens: jnp.ndarray) -> jnp.ndarray:
    """
    Check if a queen is endangered (attacked by at least one other queen).
    
    Args:
        queen_idx: Index of queen to check
        queens: All queen positions
        
    Returns:
        1 if endangered, 0 otherwise
    """
    pos = queens[queen_idx]
    
    def check_one(i):
        is_self = (i == queen_idx)
        attacks = check_attack_jit(pos, queens[i])
        return jnp.where(is_self, 0, attacks.astype(jnp.int32))
    
    attacks = jax.vmap(check_one)(jnp.arange(len(queens)))
    return jnp.minimum(jnp.sum(attacks), 1)  # 1 if any attack, 0 otherwise


@jax.jit
def compute_endangered_energy(queens: jnp.ndarray) -> jnp.ndarray:
    """
    Compute total endangered energy (number of queens under attack).
    
    Args:
        queens: All queen positions
        
    Returns:
        Number of endangered queens
    """
    endangered = jax.vmap(lambda i: is_queen_endangered(i, queens))(jnp.arange(len(queens)))
    return jnp.sum(endangered)


@jax.jit
def compute_new_energy_endangered(
    queen_idx: jnp.ndarray,
    old_pos: jnp.ndarray,
    new_pos: jnp.ndarray,
    queens: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute new endangered energy after moving a queen (O(N²) complexity).
    
    Args:
        queen_idx: Index of queen being moved
        old_pos: Old position of queen
        new_pos: New position of queen
        queens: All queen positions (before move)
        
    Returns:
        New endangered energy value
    """
    # Create new queens array with the move
    new_queens = queens.at[queen_idx].set(new_pos)
    return compute_endangered_energy(new_queens)


@jax.jit
def count_attacks_for_position(pos: jnp.ndarray, queens: jnp.ndarray, exclude_idx: int) -> jnp.ndarray:
    """
    Count how many queens attack a given position (JIT-compiled).
    
    Args:
        pos: Position to check
        queens: All queen positions
        exclude_idx: Index of queen to exclude from check
        
    Returns:
        Number of attacking queens
    """
    def check_one(i, q):
        is_excluded = (i == exclude_idx)
        attacks = check_attack_jit(pos, q)
        return jnp.where(is_excluded, 0, attacks.astype(jnp.int32))
    
    attacks = jax.vmap(lambda i: check_one(i, queens[i]))(jnp.arange(len(queens)))
    return jnp.sum(attacks)


@jax.jit
def compute_new_energy_iter(
    queen_idx: jnp.ndarray,
    old_pos: jnp.ndarray,
    new_pos: jnp.ndarray,
    old_energy: jnp.ndarray,
    queens: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute new energy after moving a queen (O(N²) complexity).
    
    Args:
        queen_idx: Index of queen being moved
        old_pos: Old position of queen
        new_pos: New position of queen
        old_energy: Current energy
        queens: All queen positions
        
    Returns:
        New energy value
    """
    old_attacks = count_attacks_for_position(old_pos, queens, queen_idx)
    new_attacks = count_attacks_for_position(new_pos, queens, queen_idx)
    return old_energy + new_attacks - old_attacks


# =============================================================================
# JIT-Compiled MCMC Step Functions - HASH (O(1))
# =============================================================================

@partial(jax.jit, static_argnums=(1,))
def mcmc_step_full_hash(
    state: Tuple,
    N: int,
    key: jnp.ndarray,
    beta: jnp.ndarray
) -> Tuple[Tuple, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Single MCMC step for full state space - O(1) complexity using hash-based line counting.
    
    Args:
        state: (queens, board, energy, line_counts)
        N: Board dimension
        key: JAX random key
        beta: Inverse temperature
        
    Returns:
        (new_state, delta_energy, accepted, new_key)
    """
    queens, board, energy, line_counts = state
    
    # Select random queen
    key, subkey = jax.random.split(key)
    queen_idx = jax.random.randint(subkey, (), 0, N**2)
    old_pos = queens[queen_idx]
    
    # Select random cell
    key, subkey = jax.random.split(key)
    new_pos = jax.random.randint(subkey, (3,), 0, N)
    
    # Check conditions
    is_self_move = (old_pos[0] == new_pos[0]) & (old_pos[1] == new_pos[1]) & (old_pos[2] == new_pos[2])
    is_occupied = board[new_pos[0], new_pos[1], new_pos[2]]
    
    # Compute energy delta (O(1))
    delta_J, proposed_line_counts = compute_energy_delta_and_update_lines(
        line_counts, old_pos, new_pos, N
    )
    
    # Mask invalid moves
    delta_J = jnp.where(is_occupied | is_self_move, 0.0, delta_J.astype(jnp.float32))
    
    # Metropolis-Hastings acceptance
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey)
    accept_condition = (delta_J <= 0) | (u < jnp.exp(-beta * delta_J))
    
    # Final acceptance (accept self-moves, reject occupied)
    accept = jnp.where(is_self_move, True, accept_condition & ~is_occupied)
    
    # Update state conditionally
    new_queens = jnp.where(accept, queens.at[queen_idx].set(new_pos), queens)
    
    new_board = jnp.where(
        accept,
        board.at[old_pos[0], old_pos[1], old_pos[2]].set(False)
             .at[new_pos[0], new_pos[1], new_pos[2]].set(True),
        board
    )
    
    new_energy = jnp.where(accept, energy + delta_J, energy)
    
    new_line_counts = jax.tree.map(
        lambda p, o: jnp.where(accept, p, o),
        proposed_line_counts,
        line_counts
    )
    
    new_state = (new_queens, new_board, new_energy, new_line_counts)
    
    return new_state, delta_J, accept, key


@partial(jax.jit, static_argnums=(1,))
def mcmc_step_reduced_hash(
    state: Tuple,
    N: int,
    key: jnp.ndarray,
    beta: jnp.ndarray
) -> Tuple[Tuple, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Single MCMC step for reduced state space - O(1) complexity using hash-based line counting.
    
    Args:
        state: (k_config, energy, line_counts)
        N: Board dimension
        key: JAX random key
        beta: Inverse temperature
        
    Returns:
        (new_state, delta_energy, accepted, new_key)
    """
    k_config, energy, line_counts = state
    
    # Select random (i, j)
    key, subkey = jax.random.split(key)
    i = jax.random.randint(subkey, (), 0, N)
    key, subkey = jax.random.split(key)
    j = jax.random.randint(subkey, (), 0, N)
    
    k_old = k_config[i, j]
    
    # Select new k (different from current)
    key, subkey = jax.random.split(key)
    offset = jax.random.randint(subkey, (), 1, N)
    k_new = (k_old + offset) % N
    
    is_self_move = (k_old == k_new)
    
    # Compute energy delta (O(1))
    delta_J, proposed_line_counts = compute_reduced_energy_delta_and_update_lines(
        line_counts, i, j, k_old, k_new, N
    )
    
    delta_J = jnp.where(is_self_move, 0.0, delta_J.astype(jnp.float32))
    
    # Metropolis-Hastings acceptance
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey)
    accept_condition = (delta_J <= 0) | (u < jnp.exp(-beta * delta_J))
    
    accept = jnp.where(is_self_move, True, accept_condition)
    
    # Update state conditionally
    new_k_config = jnp.where(accept, k_config.at[i, j].set(k_new), k_config)
    new_energy = jnp.where(accept, energy + delta_J, energy)
    
    new_line_counts = jax.tree.map(
        lambda p, o: jnp.where(accept, p, o),
        proposed_line_counts,
        line_counts
    )
    
    new_state = (new_k_config, new_energy, new_line_counts)
    
    return new_state, delta_J, accept, key


@partial(jax.jit, static_argnums=(1, 4))
def mcmc_step_reduced_iter(
    state: Tuple,
    N: int,
    key: jnp.ndarray,
    beta: jnp.ndarray,
    energy_treatment: Optional[Callable] = None
) -> Tuple[Tuple, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Single MCMC step for reduced state space - O(N²) complexity using iterative attack checking.
    
    Args:
        state: (k_config, energy, energy_untreated, queens)
        N: Board dimension
        key: JAX random key
        beta: Inverse temperature
        energy_treatment: Optional energy transformation
        
    Returns:
        (new_state, delta_energy, accepted, new_key)
    """
    k_config, energy, energy_untreated, queens = state
    
    # Select random (i, j)
    key, subkey = jax.random.split(key)
    i = jax.random.randint(subkey, (), 0, N)
    key, subkey = jax.random.split(key)
    j = jax.random.randint(subkey, (), 0, N)
    
    k_old = k_config[i, j]
    
    # Select new k (different from current)
    key, subkey = jax.random.split(key)
    offset = jax.random.randint(subkey, (), 1, N)
    k_new = (k_old + offset) % N
    
    is_self_move = (k_old == k_new)
    
    # Queen index for this (i,j) position
    queen_idx = i * N + j
    old_pos = jnp.array([i, j, k_old])
    new_pos = jnp.array([i, j, k_new])
    
    # Compute new energy (O(N²))
    new_energy_untreated = compute_new_energy_iter(queen_idx, old_pos, new_pos, energy_untreated, queens)
    
    # Compute treated delta
    delta_J_treated = compute_delta_energy_treated(
        energy_untreated, new_energy_untreated, energy_treatment
    )
    
    delta_J_treated = jnp.where(is_self_move, 0.0, delta_J_treated.astype(jnp.float32))
    
    # Metropolis-Hastings acceptance
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey)
    accept_condition = (delta_J_treated <= 0) | (u < jnp.exp(-beta * delta_J_treated))
    
    accept = jnp.where(is_self_move, True, accept_condition)
    
    # Update state conditionally
    new_k_config = jnp.where(accept, k_config.at[i, j].set(k_new), k_config)
    new_energy = jnp.where(accept, energy + delta_J_treated, energy)
    new_energy_untreated = jnp.where(accept, new_energy_untreated, energy_untreated)
    
    # Update queens array
    new_queens = jnp.where(accept, queens.at[queen_idx].set(new_pos), queens)
    
    new_state = (new_k_config, new_energy, new_energy_untreated, new_queens)
    
    return new_state, delta_J_treated, accept, key


# =============================================================================
# JIT-Compiled MCMC Step Functions - ITER (O(N²))
# =============================================================================

@partial(jax.jit, static_argnums=(1, 4))
def mcmc_step_full_iter(
    state: Tuple,
    N: int,
    key: jnp.ndarray,
    beta: jnp.ndarray,
    energy_treatment: Optional[Callable] = None
) -> Tuple[Tuple, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Single MCMC step for full state space - O(N²) complexity using iterative attack checking.
    
    Args:
        state: (queens, board, energy, energy_untreated)
        N: Board dimension
        key: JAX random key
        beta: Inverse temperature
        energy_treatment: Optional energy transformation
        
    Returns:
        (new_state, delta_energy, accepted, new_key)
    """
    queens, board, energy, energy_untreated = state
    
    # Select random queen
    key, subkey = jax.random.split(key)
    queen_idx = jax.random.randint(subkey, (), 0, N**2)
    old_pos = queens[queen_idx]
    
    # Select random cell
    key, subkey = jax.random.split(key)
    new_pos = jax.random.randint(subkey, (3,), 0, N)
    
    # Check conditions
    is_self_move = (old_pos[0] == new_pos[0]) & (old_pos[1] == new_pos[1]) & (old_pos[2] == new_pos[2])
    is_occupied = board[new_pos[0], new_pos[1], new_pos[2]]
    
    # Compute new energy (O(N²))
    new_energy_untreated = compute_new_energy_iter(queen_idx, old_pos, new_pos, energy_untreated, queens)
    
    # Compute treated delta
    delta_J_treated = compute_delta_energy_treated(
        energy_untreated, new_energy_untreated, energy_treatment
    )
    
    # Mask invalid moves
    delta_J_treated = jnp.where(
        is_occupied | is_self_move, 0.0, delta_J_treated.astype(jnp.float32)
    )
    
    # Metropolis-Hastings acceptance
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey)
    accept_condition = (delta_J_treated <= 0) | (u < jnp.exp(-beta * delta_J_treated))
    
    # Final acceptance (accept self-moves, reject occupied)
    accept = jnp.where(is_self_move, True, accept_condition & ~is_occupied)
    
    # Update state conditionally
    new_queens = jnp.where(accept, queens.at[queen_idx].set(new_pos), queens)
    
    new_board = jnp.where(
        accept,
        board.at[old_pos[0], old_pos[1], old_pos[2]].set(False)
             .at[new_pos[0], new_pos[1], new_pos[2]].set(True),
        board
    )
    
    new_energy = jnp.where(accept, energy + delta_J_treated, energy)
    new_energy_untreated = jnp.where(accept, new_energy_untreated, energy_untreated)
    
    new_state = (new_queens, new_board, new_energy, new_energy_untreated)
    
    return new_state, delta_J_treated, accept, key


# =============================================================================
# JIT-Compiled MCMC Step Functions - ENDANGERED (O(N²))
# =============================================================================

@partial(jax.jit, static_argnums=(1, 4))
def mcmc_step_full_endangered(
    state: Tuple,
    N: int,
    key: jnp.ndarray,
    beta: jnp.ndarray,
    energy_treatment: Optional[Callable] = None
) -> Tuple[Tuple, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Single MCMC step for full state space - O(N²) complexity using endangered queen counting.
    
    Args:
        state: (queens, board, energy, energy_untreated)
        N: Board dimension
        key: JAX random key
        beta: Inverse temperature
        energy_treatment: Optional energy transformation
        
    Returns:
        (new_state, delta_energy, accepted, new_key)
    """
    queens, board, energy, energy_untreated = state
    
    # Select random queen
    key, subkey = jax.random.split(key)
    queen_idx = jax.random.randint(subkey, (), 0, N**2)
    old_pos = queens[queen_idx]
    
    # Select random cell
    key, subkey = jax.random.split(key)
    new_pos = jax.random.randint(subkey, (3,), 0, N)
    
    # Check conditions
    is_self_move = (old_pos[0] == new_pos[0]) & (old_pos[1] == new_pos[1]) & (old_pos[2] == new_pos[2])
    is_occupied = board[new_pos[0], new_pos[1], new_pos[2]]
    
    # Compute new energy (O(N²)) - counts endangered queens
    new_energy_untreated = compute_new_energy_endangered(queen_idx, old_pos, new_pos, queens)
    
    # Compute treated delta
    delta_J_treated = compute_delta_energy_treated(
        energy_untreated, new_energy_untreated, energy_treatment
    )
    
    # Mask invalid moves
    delta_J_treated = jnp.where(
        is_occupied | is_self_move, 0.0, delta_J_treated.astype(jnp.float32)
    )
    
    # Metropolis-Hastings acceptance
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey)
    accept_condition = (delta_J_treated <= 0) | (u < jnp.exp(-beta * delta_J_treated))
    
    # Final acceptance (accept self-moves, reject occupied)
    accept = jnp.where(is_self_move, True, accept_condition & ~is_occupied)
    
    # Update state conditionally
    new_queens = jnp.where(accept, queens.at[queen_idx].set(new_pos), queens)
    
    new_board = jnp.where(
        accept,
        board.at[old_pos[0], old_pos[1], old_pos[2]].set(False)
             .at[new_pos[0], new_pos[1], new_pos[2]].set(True),
        board
    )
    
    new_energy = jnp.where(accept, energy + delta_J_treated, energy)
    new_energy_untreated = jnp.where(accept, new_energy_untreated, energy_untreated)
    
    new_state = (new_queens, new_board, new_energy, new_energy_untreated)
    
    return new_state, delta_J_treated, accept, key


@partial(jax.jit, static_argnums=(1, 4))
def mcmc_step_reduced_endangered(
    state: Tuple,
    N: int,
    key: jnp.ndarray,
    beta: jnp.ndarray,
    energy_treatment: Optional[Callable] = None
) -> Tuple[Tuple, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Single MCMC step for reduced state space - O(N²) complexity using endangered queen counting.
    
    Args:
        state: (k_config, energy, energy_untreated, queens)
        N: Board dimension
        key: JAX random key
        beta: Inverse temperature
        energy_treatment: Optional energy transformation
        
    Returns:
        (new_state, delta_energy, accepted, new_key)
    """
    k_config, energy, energy_untreated, queens = state
    
    # Select random (i, j)
    key, subkey = jax.random.split(key)
    i = jax.random.randint(subkey, (), 0, N)
    key, subkey = jax.random.split(key)
    j = jax.random.randint(subkey, (), 0, N)
    
    k_old = k_config[i, j]
    
    # Select new k (different from current)
    key, subkey = jax.random.split(key)
    offset = jax.random.randint(subkey, (), 1, N)
    k_new = (k_old + offset) % N
    
    is_self_move = (k_old == k_new)
    
    # Queen index for this (i,j) position
    queen_idx = i * N + j
    old_pos = jnp.array([i, j, k_old])
    new_pos = jnp.array([i, j, k_new])
    
    # Compute new energy (O(N²)) - counts endangered queens
    new_energy_untreated = compute_new_energy_endangered(queen_idx, old_pos, new_pos, queens)
    
    # Compute treated delta
    delta_J_treated = compute_delta_energy_treated(
        energy_untreated, new_energy_untreated, energy_treatment
    )
    
    delta_J_treated = jnp.where(is_self_move, 0.0, delta_J_treated.astype(jnp.float32))
    
    # Metropolis-Hastings acceptance
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey)
    accept_condition = (delta_J_treated <= 0) | (u < jnp.exp(-beta * delta_J_treated))
    
    accept = jnp.where(is_self_move, True, accept_condition)
    
    # Update state conditionally
    new_k_config = jnp.where(accept, k_config.at[i, j].set(k_new), k_config)
    new_energy = jnp.where(accept, energy + delta_J_treated, energy)
    new_energy_untreated = jnp.where(accept, new_energy_untreated, energy_untreated)
    
    # Update queens array
    new_queens = jnp.where(accept, queens.at[queen_idx].set(new_pos), queens)
    
    new_state = (new_k_config, new_energy, new_energy_untreated, new_queens)
    
    return new_state, delta_J_treated, accept, key


# =============================================================================
# Solver Base Class
# =============================================================================

class BaseSolver(SolverInterface):
    """Base class for MCMC solvers with common functionality."""
    
    def __init__(self, board: BoardInterface, seed: int = 42):
        self._board = board
        self._seed = seed
        self._key = jax.random.PRNGKey(seed)
    
    def get_board(self) -> BoardInterface:
        return self._board
    
    def set_seed(self, seed: int) -> None:
        self._seed = seed
        self._key = jax.random.PRNGKey(seed)
    
    def _get_beta(
        self,
        step: int,
        num_steps: int,
        initial_beta: float,
        final_beta: float,
        cooling: str,
        cooling_rate: float,
        simulated_annealing: bool
    ) -> float:
        """Compute beta for current step."""
        if not simulated_annealing:
            return initial_beta
        
        if cooling == 'linear':
            return compute_beta_linear(step, num_steps, initial_beta, final_beta)
        elif cooling == 'geometric':
            return compute_beta_geometric(step, cooling_rate, initial_beta)
        else:
            return compute_beta_linear(step, num_steps, initial_beta, final_beta)
    
    def _print_progress(
        self,
        step: int,
        num_steps: int,
        energy: float,
        best_energy: float,
        accept_rate: float,
        beta: float,
        elapsed: float,
        print_interval: int
    ) -> None:
        """Print progress if at print interval."""
        if step % print_interval == 0 or step == num_steps - 1:
            print(f"Step {step+1:>7}/{num_steps}: "
                  f"E={energy:>6.1f}, "
                  f"Best={best_energy:>4.0f}, "
                  f"Acc={accept_rate:>5.1%}, "
                  f"β={beta:>6.2f}, "
                  f"Time={elapsed:>5.1f}s")


# =============================================================================
# Full State Space Solver
# =============================================================================

class FullStateSolver(BaseSolver):
    """
    MCMC solver for full state space.
    
    Supports:
    - hash: O(1) energy updates using line counting
    - iter: O(N²) energy updates using iterative attack checking
    """
    
    def __init__(self, board: FullBoardState, seed: int = 42):
        if not isinstance(board, FullBoardState):
            raise TypeError("FullStateSolver requires FullBoardState")
        super().__init__(board, seed)
    
    def run(
        self,
        num_steps: int,
        initial_beta: float,
        final_beta: float,
        cooling: str,
        simulated_annealing: bool,
        energy_treatment: str = 'linear',
        complexity: str = 'hash',
        energy_reground_interval: int = 0,
        log_interval: int = 0,
        verbose: bool = True
    ) -> Tuple[FullBoardState, np.ndarray, float]:
        """
        Run MCMC optimization.
        
        Args:
            num_steps: Number of MCMC steps
            initial_beta: Starting inverse temperature
            final_beta: Final inverse temperature
            cooling: 'linear' or 'geometric'
            simulated_annealing: Whether to use annealing
            energy_treatment: 'linear', 'quadratic', 'log', 'log_quadratic'
            complexity: 'hash' for O(1) or 'iter' for O(N²)
            energy_reground_interval: Steps between energy recalculations (0 to disable)
            log_interval: Steps between progress logs (0 = auto, default 10% of steps)
            verbose: Whether to print progress
            
        Returns:
            (board, energy_history, accept_rate)
        """
        N = self._board.N
        
        # Initialize state based on complexity
        queens = self._board.get_queens().astype(jnp.int32)
        board = self._board.get_board()
        
        if complexity == 'hash':
            line_counts = self._board.get_line_counts()
            energy = jnp.array(compute_energy_from_line_counts(line_counts), dtype=jnp.float32)
            state = (queens, board, energy, line_counts)
            mcmc_step_fn = mcmc_step_full_hash
        elif complexity == 'iter':
            # iter mode - counts attacking pairs
            treatment_fn = get_energy_treatment(energy_treatment)
            energy_untreated = jnp.array(float(self._board.energy), dtype=jnp.float32)
            energy = energy_untreated if treatment_fn is None else treatment_fn(energy_untreated)
            state = (queens, board, energy, energy_untreated)
            mcmc_step_fn = mcmc_step_full_iter
        else:
            # endangered mode - counts endangered queens
            treatment_fn = get_energy_treatment(energy_treatment)
            energy_untreated = compute_endangered_energy(queens)
            energy = energy_untreated if treatment_fn is None else treatment_fn(energy_untreated)
            state = (queens, board, energy, energy_untreated)
            mcmc_step_fn = mcmc_step_full_endangered
        
        # Best state tracking
        best_queens = queens
        best_board = board
        best_energy = float(state[2]) if complexity == 'hash' else float(state[3])
        best_line_counts = line_counts if complexity == 'hash' else None
        
        energy_history = [best_energy]
        accepted_count = 0
        
        cooling_rate = compute_cooling_rate(num_steps, initial_beta, final_beta) if cooling == 'geometric' else 1.0
        
        if verbose:
            print("=" * 60)
            print(f"MCMC Solver - Full State Space (N={N})")
            print("=" * 60)
            print(f"Board: {N}×{N}×{N}, Queens: {N**2}")
            print(f"Steps: {num_steps}, Cooling: {cooling}, Complexity: {complexity}")
            print(f"β: {initial_beta} → {final_beta}" if simulated_annealing else f"β: {initial_beta}")
            print(f"Initial energy: {best_energy:.0f}")
            print("=" * 60)
        
        start_time = time.time()
        print_interval = log_interval if log_interval > 0 else max(1, num_steps // 10)
        key = self._key
        solution_found_step = None
        
        for step in range(num_steps):
            beta = self._get_beta(
                step, num_steps, initial_beta, final_beta,
                cooling, cooling_rate, simulated_annealing
            )
            beta_jnp = jnp.array(beta, dtype=jnp.float32)
            
            if complexity == 'hash':
                state, delta, accepted, key = mcmc_step_fn(state, N, key, beta_jnp)
                queens, board, energy, line_counts = state
                current_energy = float(energy)
            else:
                state, delta, accepted, key = mcmc_step_fn(state, N, key, beta_jnp, treatment_fn)
                queens, board, energy, energy_untreated = state
                current_energy = float(energy_untreated)
            
            accepted_count += int(accepted)
            
            # Track best
            if current_energy < best_energy:
                best_energy = current_energy
                best_queens = queens
                best_board = board
                if complexity == 'hash':
                    best_line_counts = line_counts
            
            # Energy regrounding (hash only)
            if complexity == 'hash' and energy_reground_interval > 0 and step % energy_reground_interval == 0 and step > 0:
                recalc = float(compute_energy_from_line_counts(line_counts))
                if abs(recalc - current_energy) > 0.01 and verbose:
                    print(f"  WARNING: Energy drift at step {step}: {current_energy:.1f} → {recalc:.1f}")
                energy = jnp.array(recalc, dtype=jnp.float32)
                state = (queens, board, energy, line_counts)
            
            # Record history
            if num_steps <= 10000 or step % 100 == 0:
                energy_history.append(current_energy)
            
            # Progress
            if verbose:
                self._print_progress(
                    step, num_steps, current_energy, best_energy,
                    accepted_count / (step + 1), beta, time.time() - start_time, print_interval
                )
            
            # Solution check
            if current_energy == 0 and solution_found_step is None:
                solution_found_step = step + 1
                if verbose:
                    print(f"\n*** SOLUTION FOUND at step {step+1}! ***")
        
        # Update board with best state
        if complexity == 'hash':
            self._board.update_state(best_queens, best_board, best_line_counts, best_energy)
        else:
            # Recompute line counts for best state
            best_line_counts = initialize_line_counts_from_queens(best_queens, N)
            self._board.update_state(best_queens, best_board, best_line_counts, best_energy)
        
        elapsed = time.time() - start_time
        accept_rate = accepted_count / num_steps
        
        if verbose:
            print("=" * 60)
            print(f"Completed in {elapsed:.1f}s ({num_steps/elapsed:.0f} steps/s)")
            print(f"Best energy: {best_energy:.0f}")
            print(f"Acceptance rate: {accept_rate:.1%}")
            if solution_found_step:
                print(f"Solution found at step: {solution_found_step}")
            print("=" * 60)
        
        return self._board, np.array(energy_history), accept_rate


# =============================================================================
# Reduced State Space Solver
# =============================================================================

class ReducedStateSolver(BaseSolver):
    """
    MCMC solver for reduced state space.
    
    Supports:
    - hash: O(1) energy updates using line counting
    - iter: O(N²) energy updates using iterative attack checking
    
    Proposal: Change k-coordinate for random (i,j) position.
    """
    
    def __init__(self, board: ReducedBoardState, seed: int = 42):
        if not isinstance(board, ReducedBoardState):
            raise TypeError("ReducedStateSolver requires ReducedBoardState")
        super().__init__(board, seed)
    
    def run(
        self,
        num_steps: int,
        initial_beta: float,
        final_beta: float,
        cooling: str,
        simulated_annealing: bool,
        energy_treatment: str = 'linear',
        complexity: str = 'hash',
        energy_reground_interval: int = 0,
        log_interval: int = 0,
        verbose: bool = True
    ) -> Tuple[ReducedBoardState, np.ndarray, float]:
        """Run MCMC optimization."""
        
        N = self._board.N
        treatment_fn = get_energy_treatment(energy_treatment)
        
        # Initialize state based on complexity
        k_config = self._board.get_k_config().astype(jnp.int32)
        
        if complexity == 'hash':
            line_counts = self._board.get_line_counts()
            energy = jnp.array(compute_energy_from_line_counts(line_counts), dtype=jnp.float32)
            state = (k_config, energy, line_counts)
            mcmc_step_fn = mcmc_step_reduced_hash
        elif complexity == 'iter':
            # iter mode - counts attacking pairs
            queens = self._board.get_queens().astype(jnp.int32)
            energy_untreated = jnp.array(float(self._board.energy), dtype=jnp.float32)
            energy = energy_untreated if treatment_fn is None else treatment_fn(energy_untreated)
            state = (k_config, energy, energy_untreated, queens)
            mcmc_step_fn = mcmc_step_reduced_iter
        else:
            # endangered mode - counts endangered queens
            queens = self._board.get_queens().astype(jnp.int32)
            energy_untreated = compute_endangered_energy(queens)
            energy = energy_untreated if treatment_fn is None else treatment_fn(energy_untreated)
            state = (k_config, energy, energy_untreated, queens)
            mcmc_step_fn = mcmc_step_reduced_endangered
        
        # Best state tracking
        best_k_config = k_config
        best_line_counts = self._board.get_line_counts() if complexity == 'hash' else None
        best_energy = float(state[2]) if complexity in ['iter', 'endangered'] else float(state[1])
        
        energy_history = [best_energy]
        accepted_count = 0
        
        cooling_rate = compute_cooling_rate(num_steps, initial_beta, final_beta) if cooling == 'geometric' else 1.0
        
        if verbose:
            print("=" * 60)
            print(f"MCMC Solver - Reduced State Space (N={N})")
            print("=" * 60)
            print(f"Board: {N}×{N}×{N}, Queens: {N**2}")
            print(f"State space: N^(N²) = {N}^{N**2}")
            print(f"Steps: {num_steps}, Cooling: {cooling}, Complexity: {complexity}")
            print(f"β: {initial_beta} → {final_beta}" if simulated_annealing else f"β: {initial_beta}")
            print(f"Initial energy: {best_energy:.0f}")
            print("=" * 60)
        
        start_time = time.time()
        print_interval = log_interval if log_interval > 0 else max(1, num_steps // 10)
        key = self._key
        solution_found_step = None
        
        for step in range(num_steps):
            beta = self._get_beta(
                step, num_steps, initial_beta, final_beta,
                cooling, cooling_rate, simulated_annealing
            )
            beta_jnp = jnp.array(beta, dtype=jnp.float32)
            
            if complexity == 'hash':
                state, delta, accepted, key = mcmc_step_fn(state, N, key, beta_jnp)
                k_config, energy, line_counts = state
                current_energy = float(energy)
            else:
                state, delta, accepted, key = mcmc_step_fn(state, N, key, beta_jnp, treatment_fn)
                k_config, energy, energy_untreated, queens = state
                current_energy = float(energy_untreated)
            
            accepted_count += int(accepted)
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_k_config = k_config
                if complexity == 'hash':
                    best_line_counts = line_counts
            
            # Energy regrounding (hash only)
            if complexity == 'hash' and energy_reground_interval > 0 and step % energy_reground_interval == 0 and step > 0:
                recalc = float(compute_energy_from_line_counts(line_counts))
                if abs(recalc - current_energy) > 0.01 and verbose:
                    print(f"  WARNING: Energy drift at step {step}: {current_energy:.1f} → {recalc:.1f}")
                energy = jnp.array(recalc, dtype=jnp.float32)
                state = (k_config, energy, line_counts)
            
            if num_steps <= 10000 or step % 100 == 0:
                energy_history.append(current_energy)
            
            if verbose:
                self._print_progress(
                    step, num_steps, current_energy, best_energy,
                    accepted_count / (step + 1), beta, time.time() - start_time, print_interval
                )
            
            if current_energy == 0 and solution_found_step is None:
                solution_found_step = step + 1
                if verbose:
                    print(f"\n*** SOLUTION FOUND at step {step+1}! ***")
        
        # Update board with best state
        if complexity == 'hash':
            self._board.update_state(best_k_config, best_line_counts, best_energy)
        else:
            # Recompute line counts for best state
            best_line_counts = initialize_line_counts_from_queens(
                self._board.get_queens(), N
            )
            self._board.update_state(best_k_config, best_line_counts, best_energy)
        
        elapsed = time.time() - start_time
        accept_rate = accepted_count / num_steps
        
        if verbose:
            print("=" * 60)
            print(f"Completed in {elapsed:.1f}s ({num_steps/elapsed:.0f} steps/s)")
            print(f"Best energy: {best_energy:.0f}")
            print(f"Acceptance rate: {accept_rate:.1%}")
            if solution_found_step:
                print(f"Solution found at step: {solution_found_step}")
            print("=" * 60)
        
        return self._board, np.array(energy_history), accept_rate


# =============================================================================
# Factory Function
# =============================================================================

def create_solver(
    board: BoardInterface,
    seed: int = 42
) -> BaseSolver:
    """Factory function to create appropriate solver for board type."""
    if isinstance(board, FullBoardState):
        return FullStateSolver(board, seed)
    elif isinstance(board, ReducedBoardState):
        return ReducedStateSolver(board, seed)
    else:
        raise TypeError(f"Unknown board type: {type(board)}")
