import jax
import jax.numpy as jnp
from functools import partial
from .board import BoardState
import time
import numpy as np

# Different energy treatments:
@jax.jit
def quadratic_energy_treatment(energy):
    return energy**2

@jax.jit
def log_energy_treatment(energy):
    return jnp.log1p(energy)

@jax.jit
def log_quadratic_energy_treatment(energy):
    return jnp.log1p(jnp.square(energy))

possible_energy_treatments = {
    'quadratic': jnp.square,
    'log': jnp.log1p,
    'log_quadratic': log_quadratic_energy_treatment
}

inverse_of_energy_treatments = {
    'quadratic': jnp.sqrt,
    'log': jnp.expm1,
    'log_quadratic': lambda x: jnp.sqrt(jnp.expm1(x))
}

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def initialize_line_counts(queens, N):
    """
    Computes the full line_counts dictionary from scratch for a given set of queens.
    This is O(N²) but only runs once or during restarts.
    """
    shapes = {
        'rook_ij': (N, N), 'rook_ik': (N, N), 'rook_jk': (N, N),
        'diag_xy1': (N, 2 * N - 1), 'diag_xy2': (N, 2 * N - 1),
        'diag_xz1': (N, 2 * N - 1), 'diag_xz2': (N, 2 * N - 1),
        'diag_yz1': (N, 2 * N - 1), 'diag_yz2': (N, 2 * N - 1),
        'space1': (3 * N - 2,), 'space2': (3 * N - 2,),
        'space3': (3 * N - 2,), 'space4': (3 * N - 2,)
    }
    
    line_counts = {k: jnp.zeros(s, dtype=jnp.int32) for k, s in shapes.items()}
    
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
    
    line_counts['space1'] = line_counts['space1'].at[Is - Js + Ks + N - 1].add(1)
    line_counts['space2'] = line_counts['space2'].at[Is - Js - Ks + 2 * (N - 1)].add(1)
    line_counts['space3'] = line_counts['space3'].at[Is + Js - Ks + N - 1].add(1)
    line_counts['space4'] = line_counts['space4'].at[Is + Js + Ks].add(1)
    
    return line_counts


# -----------------------------------------------------------------------------
# O(N²) Iterative Implementation
# -----------------------------------------------------------------------------

@jax.jit
def check_attack_jit(q1, q2):
    """
    Check if two queens q1, q2, attack each other (JIT-compiled).
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


@jax.jit
def count_attacks_for_position(pos, queens, exclude_idx):
    """
    Count how many queens attack a given position (JIT-compiled).

    Args:
        pos: given position
        queens: positions of all queens
        exclude_idx: queens I don't want to check
    """
    def check_one(i, q):
        is_excluded = (i == exclude_idx)
        attacks = check_attack_jit(pos, q)
        return jnp.where(is_excluded, 0, attacks.astype(jnp.int32))
    
    attacks = jax.vmap(lambda i: check_one(i, queens[i]))(jnp.arange(len(queens)))
    return jnp.sum(attacks)


@jax.jit
def compute_new_energy_jit(queen_idx, old_pos, new_pos, old_score_untreated, queens):
    """
    Compute energy change when moving a queen (JIT-compiled).

    Args:
        queen_idx: index of queen moved
        old_pos: old position of queen_idx
        new_pos: new position of queen_idx
        queens: all queens
    """
    old_attacks = count_attacks_for_position(old_pos, queens, queen_idx)
    new_attacks = count_attacks_for_position(new_pos, queens, queen_idx)

    return old_score_untreated + new_attacks - old_attacks

@jax.jit
def compute_delta_energy_treated(old_score_untreated, new_score_untreated, energy_treatment=None):
    if energy_treatment is not None:
        adjusted_delta = energy_treatment(new_score_untreated) - energy_treatment(old_score_untreated)
    else:
        adjusted_delta = new_score_untreated - old_score_untreated
    return adjusted_delta



@partial(jax.jit, static_argnums=(1, 4))
def mcmc_step_iter(state, N, key, beta, energy_treatment=None):
    """
    Single MCMC step - O(N²) complexity using iterative attack checking.

    Args:
        state: current state (queens, board, energy)
        N: edge of the 3d cube
        key: for random utils
        beta: temperature
        energy_treatment: optional function to adjust delta score
    
    Returns: (new_queens, new_board, new_energy), delta_J, accepted, key
    """
    queens, board, energy, energy_untreated = state
    
    # Select random queen
    key, subkey = jax.random.split(key)
    queen_idx = jax.random.randint(subkey, (), 0, N**2)
    old_pos = queens[queen_idx]
    
    # Select random cell
    key, subkey = jax.random.split(key)
    new_pos = jax.random.randint(subkey, (3,), 0, N)
    
    # Check for self-move (null move)
    is_self_move = (old_pos[0] == new_pos[0]) & (old_pos[1] == new_pos[1]) & (old_pos[2] == new_pos[2])
    
    # Check if occupied
    is_occupied = board[new_pos[0], new_pos[1], new_pos[2]]
    
    # Compute delta energy
    new_energy_untreated = compute_new_energy_jit(queen_idx, old_pos, new_pos, energy_untreated, queens) # Computes the raw energy without applying to it the function energy treatment
    delta_J_treated = compute_delta_energy_treated(energy_untreated, new_energy_untreated, energy_treatment=energy_treatment)  # Computes the delta of energy after applying the function energy treatment

    # If occupied or self-move, force delta_J_treated to 0
    delta_J_treated = jnp.where(is_occupied | is_self_move, 0.0, delta_J_treated.astype(jnp.float32))
    
    # Acceptance probability
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey)
    accept_condition = (delta_J_treated <= 0) | (u < jnp.exp(-beta * delta_J_treated))
    
    # Accept self-moves, reject occupied moves
    accept = jnp.where(is_self_move, True, accept_condition & ~is_occupied)
    
    # Update state conditionally
    new_queens = jnp.where(accept, queens.at[queen_idx].set(new_pos), queens)
    
    # Update board
    new_board = jnp.where(
        accept,
        board.at[old_pos[0], old_pos[1], old_pos[2]].set(False).at[new_pos[0], new_pos[1], new_pos[2]].set(True),
        board
    )
    
    new_energy = jnp.where(accept, energy + delta_J_treated, energy)
    new_energy_untreated = jnp.where(accept, new_energy_untreated, energy_untreated)
    
    return (new_queens, new_board, new_energy, new_energy_untreated), delta_J_treated, accept, key


# -----------------------------------------------------------------------------
# O(1) Hash-Based Implementation
# -----------------------------------------------------------------------------

def get_line_indices(i, j, k, N):
    """
    Calculates the indices for all 13 line families for a given position (i,j,k).
    Returns a dictionary matching the structure of BoardState.line_counts.
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
        'space1': i - j + k + N - 1,
        'space2': i - j - k + 2 * (N - 1),
        'space3': i + j - k + N - 1,
        'space4': i + j + k,
    }


@jax.jit
def calculate_energy_from_counts(line_counts):
    """
    Recalculates total energy from line counts.
    Energy = sum( n * (n-1) / 2 ) for every line in every family.
    """
    total_energy = 0.0
    for key in line_counts:
        counts = line_counts[key]
        pairs = (counts * (counts - 1)) / 2
        total_energy += jnp.sum(pairs)
    return total_energy


@partial(jax.jit, static_argnums=(3,))
def compute_energy_delta_and_update(line_counts, old_pos, new_pos, N):
    i_old, j_old, k_old = old_pos[0], old_pos[1], old_pos[2]
    i_new, j_new, k_new = new_pos[0], new_pos[1], new_pos[2]
    
    old_indices = get_line_indices(i_old, j_old, k_old, N)
    new_indices = get_line_indices(i_new, j_new, k_new, N)
    
    delta_J = 0.0
    new_counts_dict = {}
    
    for key in old_indices:
        idx_old = old_indices[key]
        idx_new = new_indices[key]
        counts = line_counts[key]
        c_old_val = counts[idx_old]
        c_new_val = counts[idx_new]

        # FIX: Use isinstance to check structure at trace time
        if isinstance(idx_old, tuple):
            is_same = (idx_old[0] == idx_new[0]) & (idx_old[1] == idx_new[1])
        else:
            is_same = (idx_old == idx_new)
        
        term_delta = jnp.where(is_same, 0.0, (c_new_val) - (c_old_val - 1))
        delta_J += term_delta
        
        updated_counts = counts.at[idx_old].add(-1)
        updated_counts = updated_counts.at[idx_new].add(1)
        new_counts_dict[key] = updated_counts

    return delta_J, new_counts_dict


@partial(jax.jit, static_argnums=(1,4))
def mcmc_step_hash(state, N, key, beta, energy_treatment=None):
    """
    Single MCMC step - O(1) complexity using hash-based line counting.
    
    Args:
        state: (queens, board, energy, line_counts)
        N: board size
        key: PRNG key
        beta: inverse temperature
        energy_treatment: optional function to adjust delta score

    Returns:
        (new_queens, new_board, new_energy, new_energy_untreated, new_line_counts), delta_J, accepted, key
    """
    queens, board, energy, energy_untreated, line_counts = state
    
    # Select random queen
    key, subkey = jax.random.split(key)
    queen_idx = jax.random.randint(subkey, (), 0, N**2)
    old_pos = queens[queen_idx]
    
    # Select random cell
    key, subkey = jax.random.split(key)
    new_pos = jax.random.randint(subkey, (3,), 0, N)
    
    # Check for self-move (null move)
    is_self_move = (old_pos[0] == new_pos[0]) & (old_pos[1] == new_pos[1]) & (old_pos[2] == new_pos[2])
    
    # Check if occupied
    is_occupied = board[new_pos[0], new_pos[1], new_pos[2]]
    
    # Compute delta energy and potential new line counts (O(1))
    delta_J, proposed_line_counts = compute_energy_delta_and_update(line_counts, old_pos, new_pos, N)

    #Compute new energy untreated
    new_energy_untreated = energy_untreated + delta_J

    delta_J_treated = compute_delta_energy_treated(energy_untreated, new_energy_untreated, energy_treatment=energy_treatment)  # Computes the delta of energy after applying the function energy treatment


    # If occupied or self-move, force delta_J to 0
    delta_J_treated = jnp.where(is_occupied | is_self_move, 0.0, delta_J.astype(jnp.float32))
    
    # Acceptance probability
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey)
    accept_condition = (delta_J_treated <= 0) | (u < jnp.exp(-beta * delta_J_treated))
    
    # Accept self-moves, reject occupied moves
    accept = jnp.where(is_self_move, True, accept_condition & ~is_occupied)
    
    # Update state conditionally
    new_queens = jnp.where(accept, queens.at[queen_idx].set(new_pos), queens)
    
    new_board = jnp.where(
        accept,
        board.at[old_pos[0], old_pos[1], old_pos[2]].set(False).at[new_pos[0], new_pos[1], new_pos[2]].set(True),
        board
    )
    
    new_energy = jnp.where(accept, energy + delta_J, energy)
    new_energy_untreated = jnp.where(accept, new_energy_untreated, energy_untreated)
    
    # Update line_counts: if accepted, use proposed, else keep old
    new_line_counts = jax.tree.map(
        lambda p, o: jnp.where(accept, p, o),
        proposed_line_counts,
        line_counts
    )
    
    delta_J_untreated = new_energy_untreated - energy_untreated
    return (new_queens, new_board, new_energy, new_energy_untreated, new_line_counts), delta_J_untreated, accept, key


# -----------------------------------------------------------------------------
# O(1) Hash-Based Implementation
# -----------------------------------------------------------------------------

def get_line_indices(i, j, k, N):
    """
    Calculates the indices for all 13 line families for a given position (i,j,k).
    Returns a dictionary matching the structure of BoardState.line_counts.
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
        'space1': i - j + k + N - 1,
        'space2': i - j - k + 2 * (N - 1),
        'space3': i + j - k + N - 1,
        'space4': i + j + k,
    }


@jax.jit
def calculate_energy_from_counts(line_counts):
    """
    Recalculates total energy from line counts.
    Energy = sum( n * (n-1) / 2 ) for every line in every family.
    """
    total_energy = 0.0
    for key in line_counts:
        counts = line_counts[key]
        pairs = (counts * (counts - 1)) / 2
        total_energy += jnp.sum(pairs)
    return total_energy


@partial(jax.jit, static_argnums=(3,))
def compute_energy_delta_and_update(line_counts, old_pos, new_pos, N):
    i_old, j_old, k_old = old_pos[0], old_pos[1], old_pos[2]
    i_new, j_new, k_new = new_pos[0], new_pos[1], new_pos[2]
    
    old_indices = get_line_indices(i_old, j_old, k_old, N)
    new_indices = get_line_indices(i_new, j_new, k_new, N)
    
    delta_J = 0.0
    new_counts_dict = {}
    
    for key in old_indices:
        idx_old = old_indices[key]
        idx_new = new_indices[key]
        counts = line_counts[key]
        c_old_val = counts[idx_old]
        c_new_val = counts[idx_new]

        # FIX: Use isinstance to check structure at trace time
        if isinstance(idx_old, tuple):
            is_same = (idx_old[0] == idx_new[0]) & (idx_old[1] == idx_new[1])
        else:
            is_same = (idx_old == idx_new)
        
        term_delta = jnp.where(is_same, 0.0, (c_new_val) - (c_old_val - 1))
        delta_J += term_delta
        
        updated_counts = counts.at[idx_old].add(-1)
        updated_counts = updated_counts.at[idx_new].add(1)
        new_counts_dict[key] = updated_counts

    return delta_J, new_counts_dict


@partial(jax.jit, static_argnums=(1,))
def mcmc_step_hash(state, N, key, beta):
    """
    Single MCMC step - O(1) complexity using hash-based line counting.
    
    Args:
        state: (queens, board, energy, line_counts)
        N: board size
        key: PRNG key
        beta: inverse temperature
    """
    queens, board, energy, line_counts = state
    
    # Select random queen
    key, subkey = jax.random.split(key)
    queen_idx = jax.random.randint(subkey, (), 0, N**2)
    old_pos = queens[queen_idx]
    
    # Select random cell
    key, subkey = jax.random.split(key)
    new_pos = jax.random.randint(subkey, (3,), 0, N)
    
    # Check for self-move (null move)
    is_self_move = (old_pos[0] == new_pos[0]) & (old_pos[1] == new_pos[1]) & (old_pos[2] == new_pos[2])
    
    # Check if occupied
    is_occupied = board[new_pos[0], new_pos[1], new_pos[2]]
    
    # Compute delta energy and potential new line counts (O(1))
    delta_J, proposed_line_counts = compute_energy_delta_and_update(line_counts, old_pos, new_pos, N)
    
    # If occupied or self-move, force delta_J to 0
    delta_J = jnp.where(is_occupied | is_self_move, 0.0, delta_J.astype(jnp.float32))
    
    # Acceptance probability
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey)
    accept_condition = (delta_J <= 0) | (u < jnp.exp(-beta * delta_J))
    
    # Accept self-moves, reject occupied moves
    accept = jnp.where(is_self_move, True, accept_condition & ~is_occupied)
    
    # Update state conditionally
    new_queens = jnp.where(accept, queens.at[queen_idx].set(new_pos), queens)
    
    new_board = jnp.where(
        accept,
        board.at[old_pos[0], old_pos[1], old_pos[2]].set(False).at[new_pos[0], new_pos[1], new_pos[2]].set(True),
        board
    )
    
    new_energy = jnp.where(accept, energy + delta_J, energy)
    
    # Update line_counts: if accepted, use proposed, else keep old
    new_line_counts = jax.tree.map(
        lambda p, o: jnp.where(accept, p, o),
        proposed_line_counts,
        line_counts
    )
    
    return (new_queens, new_board, new_energy, new_line_counts), delta_J, accept, key


class MCMCSolver:
    """
    MCMC solver for 3D N² Queens problem using Metropolis-Hastings.
    
    Features:
    - Multiple proposal types (move, swap, greedy)
    - Simulated annealing with adaptive/geometric cooling
    - Hash-based O(1) or iterative O(N²) energy computation
    - JIT-compiled for performance
    """
    def __init__(self, board_state):
        self.state = board_state
        self.N = board_state.N
    
    def run_improved(self, key, num_steps, initial_beta, final_beta, cooling, simulated_annealing, complexity, energy_reground_interval=0, name_energy_treatment='linear'):
        """
        Improved MCMC with multiple proposals and better cooling.
        
        Args:
            key: JAX random key
            num_steps: Number of MCMC steps
            initial_beta: Starting inverse temperature
            final_beta: Final inverse temperature
            cooling: 'linear', 'geometric', or 'adaptive'
            simulated_annealing: If False, run with constant initial_beta
            complexity: 'hash' for O(1) or 'iter' for O(N²)
            energy_reground_interval: Steps between energy recalculations (0 to disable)
        """
        N = self.N
        queens = self.state.queens.astype(jnp.int32)
        board = self.state.board
        energy_untreated = jnp.array(float(self.state.energy), dtype=jnp.float32)

        #Initial energy treatment
        if name_energy_treatment == 'linear':
            energy_treatment = None
            energy = energy_untreated
        else:
            energy_treatment = possible_energy_treatments[name_energy_treatment]
            energy = energy_treatment(energy_untreated)

        # Track best state
        best_queens = queens
        best_energy = float(energy_untreated)
        best_line_counts = None
        
        energy_history = [float(energy_untreated)]
        accepted_count = 0
        
        print("="*60)
        print(f"3D Queens MCMC Solver (N={N}) - Improved")
        print("="*60)
        print(f"Board: {N}×{N}×{N} = {N**3} cells, Queens: {N**2}")
        print(f"Steps: {num_steps}, Cooling: {cooling}, Complexity: {complexity}")
        if simulated_annealing:
            print(f"β: {initial_beta} --> {final_beta}")
        else:
            print(f"β: {initial_beta} (Constant)")
        print(f"Initial energy: {energy}")
        if energy_reground_interval > 0:
            print(f"Energy regrounding every {energy_reground_interval} steps")
        print("="*60)
        
        start_time = time.time()
        beta = initial_beta
        
        # For geometric cooling
        if cooling == 'geometric' and simulated_annealing:
            cooling_rate = (final_beta / initial_beta) ** (1.0 / num_steps)
        
        # Adaptive parameters
        window_size = 1000
        recent_accepts = []
        target_accept_rate = 0.3
        
        print_interval = max(1, num_steps // 10)
        
        solution_found_at_step = None
        
        # Select mcmc_step function based on complexity
        if complexity == 'hash':
            mcmc_step_fn = mcmc_step_hash
            line_counts = initialize_line_counts(queens, N)
        else:
            mcmc_step_fn = mcmc_step_iter
            line_counts = None
        
        for step in range(num_steps):
            if simulated_annealing:
                # Update beta based on cooling schedule
                if cooling == 'linear':
                    beta = initial_beta + (final_beta - initial_beta) * (step / num_steps)
                elif cooling == 'geometric':
                    beta = initial_beta * (cooling_rate ** step)
                elif cooling == 'adaptive':
                    # Adjust beta based on acceptance rate
                    if len(recent_accepts) >= window_size:
                        current_rate = sum(recent_accepts[-window_size:]) / window_size
                        if current_rate > target_accept_rate + 0.1:
                            beta *= 1.01  # Cool faster
                        elif current_rate < target_accept_rate - 0.1:
                            beta *= 0.99  # Heat up
                        beta = np.clip(beta, initial_beta, final_beta)
                    else:
                        beta = initial_beta + (final_beta - initial_beta) * (step / num_steps)
            else:
                beta = initial_beta
            
            state_tuple = (queens, board, energy, energy_untreated)
            beta_jnp = jnp.array(beta, dtype=jnp.float32)
            
            if complexity == 'hash':
                state_tuple = (queens, board, energy, line_counts)
                (queens, board, energy, line_counts), delta_J, accepted, key = mcmc_step_fn(
                    state_tuple, N, key, beta_jnp
                )
            else:
                state_tuple = (queens, board, energy)
                (queens, board, energy, energy_untreated), delta_J, accepted, key = mcmc_step_fn(
                    state_tuple, N, key, beta_jnp, energy_treatment=energy_treatment
                )

            accepted_count += int(accepted)
            recent_accepts.append(int(accepted))
            
            # Track best
            current_energy = float(energy)
            current_energy_untreated = float(energy_untreated)
            if current_energy_untreated < best_energy:
                best_energy = current_energy_untreated
                best_queens = queens
                if complexity == 'hash':
                    best_line_counts = line_counts
            
            # Periodic energy re-grounding
            if energy_reground_interval > 0 and step % energy_reground_interval == 0 and step > 0:
                if complexity == 'hash':
                    recalc_energy = float(calculate_energy_from_counts(line_counts))
                    if abs(recalc_energy - current_energy_untreated) > 0.01:
                        print(f"\n  WARNING: Energy drift detected at step {step}: {current_energy_untreated} -> {recalc_energy}")
                    energy = jnp.array(recalc_energy, dtype=jnp.float32)
            
            # Track energy
            if num_steps <= 10000 or step % 100 == 0:
                energy_history.append(current_energy_untreated)
            
            # Print progress
            if step % print_interval == 0 or step == num_steps - 1:
                elapsed = time.time() - start_time
                rate = (step + 1) / elapsed if elapsed > 0 else 0
                print(f"Step {step+1:>7}/{num_steps}: "
                      f"E={current_energy_untreated:>5.0f}, "
                      f"Best={best_energy:>4.0f}, "
                      f"Acc={accepted_count/(step+1):>5.1%}, "
                      f"β={beta:>6.2f}, "
                      f"{rate:>6.0f}/s")
            
            if current_energy == 0 and solution_found_at_step is None:
                solution_found_at_step = step + 1
                print(f"\nSOLUTION FOUND at step {step+1}! Continuing run...")
                
        
        # Reconstruct best state
        self.state.queens = best_queens
        self.state.board = jnp.zeros((N, N, N), dtype=bool)
        self.state.board = self.state.board.at[best_queens[:,0], best_queens[:,1], best_queens[:,2]].set(True)
        self.state.energy = jnp.array(best_energy)
        
        elapsed = time.time() - start_time
        
        print("="*60)
        print(f"Time: {elapsed:.1f}s, Rate: {(step+1)/elapsed:.0f}/s")
        print(f"Best energy: {best_energy}, Solution: {'Yes' if best_energy == 0 else 'No'}")
        if solution_found_at_step:
            print(f"First solution found at step: {solution_found_at_step}")
        print("="*60)
        
        return self.state, np.array(energy_history), accepted_count / (step + 1)
        
    def run(self, key, num_steps, initial_beta, final_beta, cooling, simulated_annealing, name_energy_treatment='linear', complexity='iter', energy_reground_interval=0):
        """
        Run MCMC with simulated annealing.
        
        Args:
            complexity: 'hash' for O(1) or 'iter' for O(N²)
            energy_reground_interval: Steps between energy recalculations (0 to disable)
        
        Optimized with JAX JIT compilation.
        """
        # Convert state to JIT-friendly format
        queens = self.state.queens.astype(jnp.int32)
        board = self.state.board
        energy_untreated = jnp.array(float(self.state.energy), dtype=jnp.float32)

        #Initial energy treatment
        if name_energy_treatment == 'linear':
            energy_treatment = None
            energy = energy_untreated
        else:
            energy_treatment = possible_energy_treatments[name_energy_treatment]
            energy = energy_treatment(energy_untreated)
        
        # Track best state
        best_queens = queens
        best_energy = float(energy_untreated)
        best_line_counts = None
        
        energy_history = [float(energy_untreated)]
        accepted_count = 0
        
        # Adaptive parameters
        stuck_count = 0
        last_energy = float(energy)
        reheat_threshold = max(500, num_steps // 20)
        restart_threshold = max(2000, num_steps // 5)
        
        print("="*60)
        print(f"3D Queens MCMC Solver (N={self.N}) - JIT Optimized")
        print("="*60)
        print(f"Board size: {self.N}×{self.N}×{self.N} = {self.N**3} cells")
        print(f"Queens: {self.N**2}")
        print(f"Steps: {num_steps}, Complexity: {complexity}")
        if simulated_annealing:
            print(f"Initial β: {initial_beta}, Final β: {final_beta}")
            print(f"Cooling: {cooling}")
            if cooling == 'adaptive':
                print(f"Adaptive: Reheat after {reheat_threshold}, Restart after {restart_threshold}")
        else:
            print(f"β: {initial_beta} (Constant)")
        print(f"Initial energy: {energy_untreated}")
        if energy_reground_interval > 0:
            print(f"Energy regrounding every {energy_reground_interval} steps")
        print("="*60)
        print("Compiling JIT functions (first run may be slow)...")
        
        start_time = time.time()
        beta = initial_beta
        
        # For geometric cooling
        if cooling == 'geometric' and simulated_annealing:
            cooling_rate = (final_beta / initial_beta) ** (1.0 / num_steps)
        
        # Print progress
        print_interval = max(1, num_steps // 1000)
        
        solution_found_at_step = None
        
        # Select mcmc_step function based on complexity
        if complexity == 'hash':
            mcmc_step_fn = mcmc_step_hash
            line_counts = initialize_line_counts(queens, board.size)
        else:
            mcmc_step_fn = mcmc_step_iter
            line_counts = None
        
        for step in range(num_steps):
            if simulated_annealing:
                if cooling == 'linear':
                    beta = initial_beta + (final_beta - initial_beta) * (step / num_steps)
                elif cooling == 'geometric':
                    beta = initial_beta * (cooling_rate ** step)
                elif cooling == 'adaptive':
                    # Standard annealing schedule base
                    scheduled_beta = initial_beta + (final_beta - initial_beta) * (step / num_steps)
                    
                    current_energy_untreated = float(energy_untreated)
                    if current_energy_untreated >= last_energy:
                        stuck_count += 1
                    else:
                        stuck_count = 0
                        last_energy = current_energy_untreated
                    
                    # Reheat if stuck
                    if stuck_count > reheat_threshold and stuck_count % reheat_threshold == 0:
                        beta = max(initial_beta, beta * 0.5)
                    
                    # Restart from best if stuck too long
                    restarted = False
                    if stuck_count > restart_threshold:
                        print(f"  >> RESTARTING at step {step+1} (Stuck {stuck_count} steps)")
                        queens = best_queens
                        board = jnp.zeros((self.N, self.N, self.N), dtype=bool)
                        board = board.at[queens[:,0], queens[:,1], queens[:,2]].set(True)
                        energy = jnp.array(best_energy, dtype=jnp.float32)
                        beta = initial_beta
                        stuck_count = 0
                        restarted = True
                        # Re-initialize line_counts from best_queens
                        if complexity == 'hash':
                            line_counts = initialize_line_counts(best_queens, N)
                    
                    # Smooth beta towards schedule (skip on restart to avoid whiplash)
                    if not restarted:
                        beta = beta + 0.1 * (scheduled_beta - beta)
                else:
                    raise ValueError(f"Unknown cooling schedule: {cooling}")
            else:
                beta = initial_beta
            
            beta_jnp = jnp.array(beta, dtype=jnp.float32)
            
            # JIT-compiled MCMC step
            if complexity == 'hash':
                state_tuple = (queens, board, energy, line_counts)
                (queens, board, energy, line_counts), delta_J, accepted, key = mcmc_step_fn(
                    state_tuple, self.N, key, beta_jnp
                )
            else:
                state_tuple = (queens, board, energy, energy_untreated)
                (queens, board, energy, energy_untreated), delta_J, accepted, key = mcmc_step_fn(
                    state_tuple, self.N, key, beta_jnp, energy_treatment=energy_treatment
                )
            accepted_count += int(accepted)
            
            # Track best
            current_energy = float(energy)
            current_energy_untreated = float(energy_untreated)
            if current_energy_untreated < best_energy:
                best_energy = current_energy_untreated
                best_queens = queens
                if complexity == 'hash':
                    best_line_counts = line_counts
            
            # Periodic energy re-grounding
            if energy_reground_interval > 0 and step % energy_reground_interval == 0 and step > 0:
                if complexity == 'hash':
                    recalc_energy = float(calculate_energy_from_counts(line_counts))
                    if abs(recalc_energy - current_energy_untreated) > 0.01:
                        print(f"\n  WARNING: Energy drift detected at step {step}: {current_energy_untreated} -> {recalc_energy}")
                    energy = jnp.array(recalc_energy, dtype=jnp.float32)
            
            # Track energy (sample every 100 steps for large runs)
            if num_steps <= 10000 or step % 100 == 0:
                energy_history.append(current_energy_untreated)
            
            # Print progress
            if step % print_interval == 0 or step == num_steps - 1:
                elapsed = time.time() - start_time
                rate = (step + 1) / elapsed if elapsed > 0 else 0
                print(f"Step {step+1:>6}/{num_steps}: "
                      f"Energy={current_energy_untreated:>6.1f}, "
                      f"Best={best_energy:>4.0f}, "
                      f"Accept={accepted_count/(step+1):>5.1%}, "
                      f"β={beta:>5.3f}, "
                      f"Rate={rate:>6.0f}/s, "
                      f"Time={elapsed:>5.1f}s")
            
            # Check for solution
            if current_energy == 0 and solution_found_at_step is None:
                solution_found_at_step = step + 1
                print(f"\nSOLUTION FOUND at step {step+1}! Continuing run...")
        
        # Reconstruct best state
        self.state.queens = best_queens
        self.state.board = jnp.zeros((self.N, self.N, self.N), dtype=bool)
        self.state.board = self.state.board.at[best_queens[:,0], best_queens[:,1], best_queens[:,2]].set(True)
        self.state.energy = jnp.array(best_energy)
        
        elapsed = time.time() - start_time
        final_accept_rate = accepted_count / (step + 1)
        
        print("="*60)
        print("MCMC Summary")
        print("="*60)
        print(f"Total time: {elapsed:.1f} seconds")
        print(f"Steps/second: {(step+1)/elapsed:.0f}")
        print(f"Best energy found: {best_energy}")
        print(f"Acceptance rate: {final_accept_rate:.2%}")
        print(f"Solution found: {'Yes' if best_energy == 0 else 'No'}")
        if solution_found_at_step:
            print(f"First solution found at step: {solution_found_at_step}")
        print("="*60)
        
        return self.state, np.array(energy_history), final_accept_rate
