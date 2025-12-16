"""
Utility functions for 3D N² Queens MCMC Solver.

This module contains:
- JIT-compiled energy computation functions
- Line index calculations
- Attack checking functions
- Energy treatment functions
- Common helper functions
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Dict, Tuple, List, Optional, Callable


# =============================================================================
# Energy Treatment Functions
# =============================================================================

@jax.jit
def identity_treatment(energy: jnp.ndarray) -> jnp.ndarray:
    """Identity (linear) energy treatment."""
    return energy


@jax.jit
def quadratic_treatment(energy: jnp.ndarray) -> jnp.ndarray:
    """Quadratic energy treatment: E² """
    return jnp.square(energy)


@jax.jit
def log_treatment(energy: jnp.ndarray) -> jnp.ndarray:
    """Logarithmic energy treatment: log(1 + E)"""
    return jnp.log1p(energy)


@jax.jit
def log_quadratic_treatment(energy: jnp.ndarray) -> jnp.ndarray:
    """Log-quadratic energy treatment: log(1 + E²)"""
    return jnp.log1p(jnp.square(energy))


ENERGY_TREATMENTS: Dict[str, Callable] = {
    'linear': None,  # None means identity (no transformation)
    'quadratic': quadratic_treatment,
    'log': log_treatment,
    'log_quadratic': log_quadratic_treatment,
}


def get_energy_treatment(name: str) -> Optional[Callable]:
    """
    Get energy treatment function by name.
    
    Args:
        name: Treatment name ('linear', 'quadratic', 'log', 'log_quadratic')
        
    Returns:
        Treatment function or None for linear.
    """
    if name not in ENERGY_TREATMENTS:
        raise ValueError(f"Unknown energy treatment: {name}. "
                        f"Valid options: {list(ENERGY_TREATMENTS.keys())}")
    return ENERGY_TREATMENTS[name]


# =============================================================================
# Attack Checking Functions
# =============================================================================

@jax.jit
def check_attack_jit(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """
    Check if two queens attack each other (JIT-compiled).
    
    Attack types:
    - Rook-type: share at least two coordinates
    - Planar diagonal: |i-i'| = |j-j'| != 0 when k=k' (and analogous)
    - Space diagonal: |i-i'| = |j-j'| = |k-k'| != 0
    
    Args:
        q1: First queen position (i, j, k)
        q2: Second queen position (i, j, k)
        
    Returns:
        Boolean indicating if queens attack each other.
    """
    i1, j1, k1 = q1[0], q1[1], q1[2]
    i2, j2, k2 = q2[0], q2[1], q2[2]
    
    di = jnp.abs(i1 - i2)
    dj = jnp.abs(j1 - j2)
    dk = jnp.abs(k1 - k2)
    
    # Rook-type: share at least two coordinates
    rook = ((i1 == i2) & (j1 == j2)) | ((i1 == i2) & (k1 == k2)) | ((j1 == j2) & (k1 == k2))
    
    # Planar diagonals
    planar_xy = (k1 == k2) & (di == dj) & (di != 0)
    planar_xz = (j1 == j2) & (di == dk) & (di != 0)
    planar_yz = (i1 == i2) & (dj == dk) & (dj != 0)
    
    # Space diagonal
    space = (di == dj) & (dj == dk) & (di != 0)
    
    return rook | planar_xy | planar_xz | planar_yz | space


def check_attack_python(q1: Tuple[int, int, int], q2: Tuple[int, int, int]) -> bool:
    """
    Check if two queens attack each other (pure Python for testing).
    
    Args:
        q1: First queen position (i, j, k)
        q2: Second queen position (i, j, k)
        
    Returns:
        Boolean indicating if queens attack each other.
    """
    i1, j1, k1 = q1
    i2, j2, k2 = q2
    
    di = abs(i1 - i2)
    dj = abs(j1 - j2)
    dk = abs(k1 - k2)
    
    # Rook-type
    if (i1 == i2 and j1 == j2) or (i1 == i2 and k1 == k2) or (j1 == j2 and k1 == k2):
        return True
    
    # Planar diagonals
    if k1 == k2 and di == dj and di != 0:
        return True
    if j1 == j2 and di == dk and di != 0:
        return True
    if i1 == i2 and dj == dk and dj != 0:
        return True
    
    # Space diagonal
    if di == dj == dk and di != 0:
        return True
    
    return False


@partial(jax.jit, static_argnums=(1,))
def compute_energy_iterative(queens: jnp.ndarray, N: int) -> jnp.ndarray:
    """
    Compute energy by iterating over all queen pairs - O(N⁴) complexity.
    
    This is the ground truth energy computation.
    
    Args:
        queens: Array of shape (N², 3) with queen positions
        N: Board dimension
        
    Returns:
        Number of attacking pairs.
    """
    num_queens = N * N
    energy = 0.0
    
    for i in range(num_queens):
        for j in range(i + 1, num_queens):
            attack = check_attack_jit(queens[i], queens[j])
            energy += jnp.where(attack, 1.0, 0.0)
    
    return energy


@partial(jax.jit, static_argnums=(1,))
def compute_new_energy_after_move(
    queen_idx: jnp.ndarray,
    old_pos: jnp.ndarray,
    new_pos: jnp.ndarray,
    current_energy: jnp.ndarray,
    queens: jnp.ndarray,
    N: int
) -> jnp.ndarray:
    """
    Compute new energy after moving a queen - O(N²) complexity.
    
    Instead of recomputing all pairs, we only check attacks involving
    the moved queen.
    
    Args:
        queen_idx: Index of queen being moved
        old_pos: Old position
        new_pos: New position
        current_energy: Current energy
        queens: All queen positions
        N: Board dimension
        
    Returns:
        New energy after the move.
    """
    num_queens = N * N
    
    # Count attacks at old position
    old_attacks = 0.0
    for i in range(num_queens):
        is_self = (i == queen_idx)
        attack = check_attack_jit(old_pos, queens[i])
        old_attacks += jnp.where(is_self, 0.0, jnp.where(attack, 1.0, 0.0))
    
    # Count attacks at new position
    new_attacks = 0.0
    for i in range(num_queens):
        is_self = (i == queen_idx)
        attack = check_attack_jit(new_pos, queens[i])
        new_attacks += jnp.where(is_self, 0.0, jnp.where(attack, 1.0, 0.0))
    
    return current_energy - old_attacks + new_attacks


def count_attacking_pairs(queens: List[Tuple[int, int, int]]) -> int:
    """
    Count attacking pairs using naive O(N⁴) algorithm.
    
    This is the ground truth for testing.
    
    Args:
        queens: List of queen positions as (i, j, k) tuples.
        
    Returns:
        Number of attacking pairs.
    """
    count = 0
    n = len(queens)
    for i in range(n):
        for j in range(i + 1, n):
            if check_attack_python(queens[i], queens[j]):
                count += 1
    return count


# =============================================================================
# Colored Energy Functions (conflicts + black square penalty)
# =============================================================================

@jax.jit
def is_black_square(pos: jnp.ndarray) -> jnp.ndarray:
    """Check if position is a black square: (i+j+k) mod 2 == 1"""
    return ((pos[0] + pos[1] + pos[2]) % 2) == 1


@jax.jit
def count_black_squares(queens: jnp.ndarray) -> jnp.ndarray:
    """Count number of queens on black squares."""
    black = jax.vmap(is_black_square)(queens)
    return jnp.sum(black)


@jax.jit
def compute_colored_energy(queens: jnp.ndarray, conflicts: jnp.ndarray) -> jnp.ndarray:
    """
    Compute colored energy: conflicts + 4 * (queens on black squares).
    
    Args:
        queens: Array of shape (N², 3) with queen positions
        conflicts: Number of attacking pairs
        
    Returns:
        Colored energy value.
    """
    black_count = count_black_squares(queens)
    return conflicts + 4.0 * black_count


def compute_colored_energy_python(queens: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    """
    Compute colored energy using Python (for verification).
    
    Args:
        queens: List of queen positions as (i, j, k) tuples.
        
    Returns:
        Tuple of (total_energy, conflicts, black_count)
    """
    conflicts = count_attacking_pairs(queens)
    black_count = sum(1 for q in queens if (q[0] + q[1] + q[2]) % 2 == 1)
    total = conflicts + 4 * black_count
    return total, conflicts, black_count


# =============================================================================
# Weighted Energy Functions (conflicts + sum of |x+y-2z|)
# =============================================================================

@jax.jit
def position_weight(pos: jnp.ndarray) -> jnp.ndarray:
    """Compute position weight: |x + y - 2z|"""
    return jnp.abs(pos[0] + pos[1] - 2 * pos[2])


@jax.jit
def compute_total_weight(queens: jnp.ndarray) -> jnp.ndarray:
    """Compute sum of |x+y-2z| for all queens."""
    weights = jax.vmap(position_weight)(queens)
    return jnp.sum(weights)


@jax.jit
def compute_weighted_energy(queens: jnp.ndarray, conflicts: jnp.ndarray) -> jnp.ndarray:
    """
    Compute weighted energy: conflicts + sum of |x+y-2z| for all queens.
    
    Args:
        queens: Array of shape (N², 3) with queen positions
        conflicts: Number of attacking pairs
        
    Returns:
        Weighted energy value.
    """
    total_weight = compute_total_weight(queens)
    return conflicts + total_weight


def compute_weighted_energy_python(queens: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    """
    Compute weighted energy using Python (for verification).
    
    Args:
        queens: List of queen positions as (i, j, k) tuples.
        
    Returns:
        Tuple of (total_energy, conflicts, total_weight)
    """
    conflicts = count_attacking_pairs(queens)
    total_weight = sum(abs(q[0] + q[1] - 2 * q[2]) for q in queens)
    total = conflicts + total_weight
    return total, conflicts, total_weight


# =============================================================================
# Colored Endangered Energy Functions (endangered + 4 * black squares)
# =============================================================================

@jax.jit
def compute_colored_endangered_energy(queens: jnp.ndarray, endangered: jnp.ndarray) -> jnp.ndarray:
    """
    Compute colored endangered energy: endangered + 4 * black_squares.
    
    Args:
        queens: Array of shape (N², 3) with queen positions
        endangered: Number of endangered queens
        
    Returns:
        Colored endangered energy value.
    """
    black_count = count_black_squares(queens)
    return endangered + 4.0 * black_count


def compute_colored_endangered_energy_python(queens: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    """
    Compute colored endangered energy using Python (for verification).
    
    Args:
        queens: List of queen positions as (i, j, k) tuples.
        
    Returns:
        Tuple of (total_energy, endangered_count, black_count)
    """
    # Count endangered queens
    n = len(queens)
    endangered_set = set()
    for i in range(n):
        for j in range(i + 1, n):
            if check_attack_python(queens[i], queens[j]):
                endangered_set.add(i)
                endangered_set.add(j)
    endangered_count = len(endangered_set)
    
    black_count = sum(1 for q in queens if (q[0] + q[1] + q[2]) % 2 == 1)
    total = endangered_count + 4 * black_count
    return total, endangered_count, black_count


# =============================================================================
# Weighted Endangered Energy Functions (endangered + sum|x+y-2z|)
# =============================================================================

@jax.jit
def compute_weighted_endangered_energy(queens: jnp.ndarray, endangered: jnp.ndarray) -> jnp.ndarray:
    """
    Compute weighted endangered energy: endangered + sum of |x+y-2z| for all queens.
    
    Args:
        queens: Array of shape (N², 3) with queen positions
        endangered: Number of endangered queens
        
    Returns:
        Weighted endangered energy value.
    """
    total_weight = compute_total_weight(queens)
    return endangered + total_weight


def compute_weighted_endangered_energy_python(queens: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    """
    Compute weighted endangered energy using Python (for verification).
    
    Args:
        queens: List of queen positions as (i, j, k) tuples.
        
    Returns:
        Tuple of (total_energy, endangered_count, total_weight)
    """
    # Count endangered queens
    n = len(queens)
    endangered_set = set()
    for i in range(n):
        for j in range(i + 1, n):
            if check_attack_python(queens[i], queens[j]):
                endangered_set.add(i)
                endangered_set.add(j)
    endangered_count = len(endangered_set)
    
    total_weight = sum(abs(q[0] + q[1] - 2 * q[2]) for q in queens)
    total = endangered_count + total_weight
    return total, endangered_count, total_weight


# =============================================================================
# Line Index Functions (for O(1) energy updates)
# =============================================================================

def get_line_indices(i: int, j: int, k: int, N: int) -> Dict[str, Tuple]:
    """
    Calculate indices for all 13 line families for a position (i, j, k).
    
    Line families:
    - 3 rook lines: (i,j), (i,k), (j,k)
    - 6 planar diagonals: 2 per plane
    - 4 space diagonals
    
    Args:
        i, j, k: Position coordinates
        N: Board dimension
        
    Returns:
        Dictionary mapping line family names to index tuples/values.
    """
    return {
        'rook_ij': (i, j),
        'rook_ik': (i, k),
        'rook_jk': (j, k),
        'diag_xy1': (k, i - j + N - 1),
        'diag_xy2': (k, i + j),
        'diag_xz1': (j, i - k + N - 1),
        'diag_xz2': (j, i + k),
        'diag_yz1': (i, j - k + N - 1),
        'diag_yz2': (i, j + k),
        # Space diagonals with 2D indices
        'space1': (i - j + N - 1, j - k + N - 1),
        'space2': (i - j + N - 1, j + k),
        'space3': (i + j, j + k),
        'space4': (i + j, j - k + N - 1),
    }


def get_line_count_shapes(N: int) -> Dict[str, Tuple[int, ...]]:
    """
    Get shapes for all line count arrays.
    
    Args:
        N: Board dimension
        
    Returns:
        Dictionary mapping line family names to array shapes.
    """
    return {
        'rook_ij': (N, N),
        'rook_ik': (N, N),
        'rook_jk': (N, N),
        'diag_xy1': (N, 2 * N - 1),
        'diag_xy2': (N, 2 * N - 1),
        'diag_xz1': (N, 2 * N - 1),
        'diag_xz2': (N, 2 * N - 1),
        'diag_yz1': (N, 2 * N - 1),
        'diag_yz2': (N, 2 * N - 1),
        # Space diagonals need 2D indices to correctly identify lines
        # space1 (+1,+1,+1): indexed by (i-j, j-k)
        # space2 (+1,+1,-1): indexed by (i-j, j+k)
        # space3 (+1,-1,+1): indexed by (i+j, j+k)
        # space4 (-1,+1,+1): indexed by (i+j, j-k)
        'space1': (2 * N - 1, 2 * N - 1),
        'space2': (2 * N - 1, 2 * N - 1),
        'space3': (2 * N - 1, 2 * N - 1),
        'space4': (2 * N - 1, 2 * N - 1),
    }


def create_empty_line_counts(N: int) -> Dict[str, jnp.ndarray]:
    """
    Create empty line count structures.
    
    Args:
        N: Board dimension
        
    Returns:
        Dictionary of zero-initialized line count arrays.
    """
    shapes = get_line_count_shapes(N)
    return {k: jnp.zeros(s, dtype=jnp.int32) for k, s in shapes.items()}


def initialize_line_counts_from_queens(queens: jnp.ndarray, N: int) -> Dict[str, jnp.ndarray]:
    """
    Initialize line counts from queen positions.
    
    Args:
        queens: Array of shape (N², 3) with queen positions
        N: Board dimension
        
    Returns:
        Dictionary of line count arrays.
    """
    line_counts = create_empty_line_counts(N)
    
    qs = queens.astype(jnp.int32)
    Is, Js, Ks = qs[:, 0], qs[:, 1], qs[:, 2]
    
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


# =============================================================================
# Energy Computation Functions
# =============================================================================

@jax.jit
def compute_energy_from_line_counts(line_counts: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """
    Compute energy from line counts.
    
    Energy = sum over all lines of n*(n-1)/2 where n is the count on that line.
    
    Note: This overcounts because pairs can attack on multiple lines.
    However, it reaches 0 exactly when the true energy is 0.
    
    Args:
        line_counts: Dictionary of line count arrays
        
    Returns:
        Total energy (surrogate).
    """
    total_energy = 0.0
    for key in line_counts:
        counts = line_counts[key]
        pairs = (counts * (counts - 1)) / 2
        total_energy += jnp.sum(pairs)
    return total_energy


@partial(jax.jit, static_argnums=(2,))
def compute_delta_energy_treated(
    old_energy: jnp.ndarray,
    new_energy: jnp.ndarray,
    energy_treatment: Optional[Callable] = None
) -> jnp.ndarray:
    """
    Compute treated energy delta.
    
    Args:
        old_energy: Old untreated energy
        new_energy: New untreated energy
        energy_treatment: Optional treatment function
        
    Returns:
        Delta of treated energies.
    """
    if energy_treatment is not None:
        return energy_treatment(new_energy) - energy_treatment(old_energy)
    return new_energy - old_energy


# =============================================================================
# MCMC Step Helper Functions
# =============================================================================

@partial(jax.jit, static_argnums=(3,))
def compute_energy_delta_and_update_lines(
    line_counts: Dict[str, jnp.ndarray],
    old_pos: jnp.ndarray,
    new_pos: jnp.ndarray,
    N: int
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Compute energy delta and update line counts for a queen move.
    
    Args:
        line_counts: Current line counts
        old_pos: Old position (i, j, k)
        new_pos: New position (i, j, k)
        N: Board dimension
        
    Returns:
        Tuple of (delta_energy, new_line_counts)
    """
    i_old, j_old, k_old = old_pos[0], old_pos[1], old_pos[2]
    i_new, j_new, k_new = new_pos[0], new_pos[1], new_pos[2]
    
    old_indices = get_line_indices(i_old, j_old, k_old, N)
    new_indices = get_line_indices(i_new, j_new, k_new, N)
    
    delta_J = 0.0
    new_counts = {}
    
    for key in old_indices:
        idx_old = old_indices[key]
        idx_new = new_indices[key]
        counts = line_counts[key]
        c_old = counts[idx_old]
        c_new = counts[idx_new]
        
        # Check if same index
        if isinstance(idx_old, tuple):
            is_same = (idx_old[0] == idx_new[0]) & (idx_old[1] == idx_new[1])
        else:
            is_same = (idx_old == idx_new)
        
        # Delta: c_new - (c_old - 1) when moving
        term_delta = jnp.where(is_same, 0.0, c_new - (c_old - 1))
        delta_J += term_delta
        
        # Update counts
        updated = counts.at[idx_old].add(-1)
        updated = updated.at[idx_new].add(1)
        new_counts[key] = updated
    
    return delta_J, new_counts


@partial(jax.jit, static_argnums=(5,))
def compute_reduced_energy_delta_and_update_lines(
    line_counts: Dict[str, jnp.ndarray],
    i: jnp.ndarray,
    j: jnp.ndarray,
    k_old: jnp.ndarray,
    k_new: jnp.ndarray,
    N: int
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Compute energy delta for reduced state space (only k changes).
    
    Args:
        line_counts: Current line counts
        i, j: Fixed position indices
        k_old: Old k-coordinate
        k_new: New k-coordinate
        N: Board dimension
        
    Returns:
        Tuple of (delta_energy, new_line_counts)
    """
    old_indices = get_line_indices(i, j, k_old, N)
    new_indices = get_line_indices(i, j, k_new, N)
    
    delta_J = 0.0
    new_counts = {}
    
    for key in old_indices:
        idx_old = old_indices[key]
        idx_new = new_indices[key]
        counts = line_counts[key]
        c_old = counts[idx_old]
        c_new = counts[idx_new]
        
        if isinstance(idx_old, tuple):
            is_same = (idx_old[0] == idx_new[0]) & (idx_old[1] == idx_new[1])
        else:
            is_same = (idx_old == idx_new)
        
        term_delta = jnp.where(is_same, 0.0, c_new - (c_old - 1))
        delta_J += term_delta
        
        updated = counts.at[idx_old].add(-1)
        updated = updated.at[idx_new].add(1)
        new_counts[key] = updated
    
    return delta_J, new_counts


# =============================================================================
# Cooling Schedule Functions
# =============================================================================

def compute_beta_linear(
    step: int,
    num_steps: int,
    initial_beta: float,
    final_beta: float
) -> float:
    """Linear cooling schedule."""
    return initial_beta + (final_beta - initial_beta) * (step / num_steps)


def compute_beta_geometric(
    step: int,
    cooling_rate: float,
    initial_beta: float
) -> float:
    """Geometric cooling schedule."""
    return initial_beta * (cooling_rate ** step)


def compute_cooling_rate(
    num_steps: int,
    initial_beta: float,
    final_beta: float
) -> float:
    """Compute geometric cooling rate."""
    return (final_beta / initial_beta) ** (1.0 / num_steps)


# =============================================================================
# Solvability Check
# =============================================================================

def check_solvability(N: int) -> Dict[str, any]:
    """
    Check if the 3D N² Queens problem is theoretically solvable.
    
    Based on Klarner's theorem (1967):
    - If gcd(N, 210) = 1, a solution exists
    - 210 = 2 × 3 × 5 × 7
    
    Args:
        N: Board dimension
        
    Returns:
        Dictionary with solvability information.
    """
    import math
    gcd = math.gcd(N, 210)
    solvable = (gcd == 1)
    
    factors = []
    if N % 2 == 0: factors.append(2)
    if N % 3 == 0: factors.append(3)
    if N % 5 == 0: factors.append(5)
    if N % 7 == 0: factors.append(7)
    
    return {
        'N': N,
        'gcd_210': gcd,
        'solvable': solvable,
        'blocking_factors': factors,
        'queens': N**2,
        'cells': N**3,
        'density': N**2 / N**3,
    }
