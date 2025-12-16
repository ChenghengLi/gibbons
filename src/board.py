"""
Board state implementations for 3D N² Queens Problem.

This module provides two board state implementations:
- FullBoardState: General state space with C(N³, N²) configurations
- ReducedBoardState: Restricted state space with N^(N²) configurations
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

from .interfaces import BoardInterface
from .utils import (
    check_attack_python,
    count_attacking_pairs,
    initialize_line_counts_from_queens,
    create_empty_line_counts,
    get_line_indices,
    compute_energy_from_line_counts,
)


class FullBoardState(BoardInterface):
    """
    Full state space board representation.
    
    In the full state space:
    - Configuration: s = {(i₁,j₁,k₁), ..., (i_{N²},j_{N²},k_{N²})}
    - State space size: C(N³, N²)
    - Queens can be placed anywhere on the N×N×N cube
    
    Attributes:
        _N: Board dimension
        _queens: Array of shape (N², 3) with queen positions
        _board: 3D boolean occupancy grid
        _line_counts: Dictionary of line count arrays
        _energy: Current energy (attacking pairs)
    """
    
    def __init__(self, key: jnp.ndarray, N: int):
        """
        Initialize board with random queen placement.
        
        Args:
            key: JAX random key
            N: Board dimension
        """
        self._N = N
        self._queen_count = N ** 2
        self._initialize_random(key)
    
    @property
    def N(self) -> int:
        return self._N
    
    @property
    def queen_count(self) -> int:
        return self._queen_count
    
    @property
    def energy(self) -> float:
        return float(self._energy)
    
    def get_queens(self) -> jnp.ndarray:
        return self._queens
    
    def get_line_counts(self) -> Dict[str, jnp.ndarray]:
        return self._line_counts
    
    def get_board(self) -> jnp.ndarray:
        """Get 3D occupancy grid."""
        return self._board
    
    def _initialize_random(self, key: jnp.ndarray) -> None:
        """Initialize with random queen placement."""
        key, subkey = jax.random.split(key)
        
        # Select N² random cells from N³ total
        total_cells = self._N ** 3
        flat_indices = jax.random.choice(
            subkey, total_cells, shape=(self._queen_count,), replace=False
        )
        
        # Convert to 3D coordinates
        self._queens = jnp.array(
            np.unravel_index(flat_indices, (self._N, self._N, self._N))
        ).T.astype(jnp.int32)
        
        # Create occupancy grid
        self._board = jnp.zeros((self._N, self._N, self._N), dtype=bool)
        self._board = self._board.at[
            self._queens[:, 0], self._queens[:, 1], self._queens[:, 2]
        ].set(True)
        
        # Initialize line counts
        self._line_counts = initialize_line_counts_from_queens(self._queens, self._N)
        
        # Compute initial energy
        self._energy = jnp.array(self.compute_energy(), dtype=jnp.float32)
    
    def compute_energy(self) -> float:
        """Compute energy using naive O(N⁴) algorithm."""
        queens_list = [
            (int(q[0]), int(q[1]), int(q[2])) 
            for q in self._queens
        ]
        return float(count_attacking_pairs(queens_list))
    
    def copy(self) -> 'FullBoardState':
        """Create a deep copy."""
        new_board = FullBoardState.__new__(FullBoardState)
        new_board._N = self._N
        new_board._queen_count = self._queen_count
        new_board._queens = self._queens.copy()
        new_board._board = self._board.copy()
        new_board._line_counts = {k: v.copy() for k, v in self._line_counts.items()}
        new_board._energy = self._energy
        return new_board
    
    def update_state(
        self,
        queens: jnp.ndarray,
        board: jnp.ndarray,
        line_counts: Dict[str, jnp.ndarray],
        energy: float
    ) -> None:
        """
        Update internal state (used by solver).
        
        Args:
            queens: New queen positions
            board: New occupancy grid
            line_counts: New line counts
            energy: New energy value
        """
        self._queens = queens
        self._board = board
        self._line_counts = line_counts
        self._energy = jnp.array(energy, dtype=jnp.float32)


class ReducedBoardState(BoardInterface):
    """
    Reduced state space board representation.
    
    In the reduced state space:
    - Configuration: k: {0,...,N-1}² → {0,...,N-1}
    - Each (i,j) pair has exactly one queen at height k(i,j)
    - State space size: N^(N²)
    - Guarantees no rook attacks in ij direction
    
    Attributes:
        _N: Board dimension
        _k_config: 2D array of shape (N, N) storing k-coordinates
        _line_counts: Dictionary of line count arrays
        _energy: Current energy (attacking pairs)
    """
    
    def __init__(self, key: jnp.ndarray, N: int):
        """
        Initialize board with random k-configuration.
        
        Args:
            key: JAX random key
            N: Board dimension
        """
        self._N = N
        self._queen_count = N ** 2
        self._initialize_random(key)
    
    @property
    def N(self) -> int:
        return self._N
    
    @property
    def queen_count(self) -> int:
        return self._queen_count
    
    @property
    def energy(self) -> float:
        return float(self._energy)
    
    def get_k_config(self) -> jnp.ndarray:
        """Get k-configuration array."""
        return self._k_config
    
    def get_queens(self) -> jnp.ndarray:
        """Derive queens from k-configuration."""
        i_coords, j_coords = jnp.meshgrid(
            jnp.arange(self._N), jnp.arange(self._N), indexing='ij'
        )
        queens = jnp.stack([
            i_coords.flatten(),
            j_coords.flatten(),
            self._k_config.flatten()
        ], axis=1)
        return queens.astype(jnp.int32)
    
    def get_line_counts(self) -> Dict[str, jnp.ndarray]:
        return self._line_counts
    
    def _initialize_random(self, key: jnp.ndarray) -> None:
        """Initialize with random k-configuration."""
        # Random k value for each (i,j)
        self._k_config = jax.random.randint(key, (self._N, self._N), 0, self._N)
        
        # Initialize line counts
        self._line_counts = self._initialize_line_counts()
        
        # Compute initial energy
        self._energy = jnp.array(self.compute_energy(), dtype=jnp.float32)
    
    def _initialize_line_counts(self) -> Dict[str, jnp.ndarray]:
        """Initialize line counts from k-configuration."""
        line_counts = create_empty_line_counts(self._N)
        N = self._N
        
        i_coords, j_coords = jnp.meshgrid(
            jnp.arange(N), jnp.arange(N), indexing='ij'
        )
        Is = i_coords.flatten().astype(jnp.int32)
        Js = j_coords.flatten().astype(jnp.int32)
        Ks = self._k_config.flatten().astype(jnp.int32)
        
        line_counts['rook_ij'] = line_counts['rook_ij'].at[Is, Js].add(1)
        line_counts['rook_ik'] = line_counts['rook_ik'].at[Is, Ks].add(1)
        line_counts['rook_jk'] = line_counts['rook_jk'].at[Js, Ks].add(1)
        
        line_counts['diag_xy1'] = line_counts['diag_xy1'].at[Ks, Is - Js + N - 1].add(1)
        line_counts['diag_xy2'] = line_counts['diag_xy2'].at[Ks, Is + Js].add(1)
        line_counts['diag_xz1'] = line_counts['diag_xz1'].at[Js, Is - Ks + N - 1].add(1)
        line_counts['diag_xz2'] = line_counts['diag_xz2'].at[Js, Is + Ks].add(1)
        line_counts['diag_yz1'] = line_counts['diag_yz1'].at[Is, Js - Ks + N - 1].add(1)
        line_counts['diag_yz2'] = line_counts['diag_yz2'].at[Is, Js + Ks].add(1)
        
        # Space diagonals with 2D indices
        # space1 (+1,+1,+1): indexed by (i-j, j-k)
        line_counts['space1'] = line_counts['space1'].at[Is - Js + N - 1, Js - Ks + N - 1].add(1)
        # space2 (+1,+1,-1): indexed by (i-j, j+k)
        line_counts['space2'] = line_counts['space2'].at[Is - Js + N - 1, Js + Ks].add(1)
        # space3 (+1,-1,+1): indexed by (i+j, j+k)
        line_counts['space3'] = line_counts['space3'].at[Is + Js, Js + Ks].add(1)
        # space4 (-1,+1,+1): indexed by (i+j, j-k)
        line_counts['space4'] = line_counts['space4'].at[Is + Js, Js - Ks + N - 1].add(1)
        
        return line_counts
    
    def compute_energy(self) -> float:
        """Compute energy using naive O(N⁴) algorithm."""
        queens = self.get_queens()
        queens_list = [
            (int(q[0]), int(q[1]), int(q[2])) 
            for q in queens
        ]
        return float(count_attacking_pairs(queens_list))
    
    def copy(self) -> 'ReducedBoardState':
        """Create a deep copy."""
        new_board = ReducedBoardState.__new__(ReducedBoardState)
        new_board._N = self._N
        new_board._queen_count = self._queen_count
        new_board._k_config = self._k_config.copy()
        new_board._line_counts = {k: v.copy() for k, v in self._line_counts.items()}
        new_board._energy = self._energy
        return new_board
    
    def update_state(
        self,
        k_config: jnp.ndarray,
        line_counts: Dict[str, jnp.ndarray],
        energy: float
    ) -> None:
        """
        Update internal state (used by solver).
        
        Args:
            k_config: New k-configuration
            line_counts: New line counts
            energy: New energy value
        """
        self._k_config = k_config
        self._line_counts = line_counts
        self._energy = jnp.array(energy, dtype=jnp.float32)
