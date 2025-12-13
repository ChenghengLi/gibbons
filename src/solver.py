"""
The file consists in two:
1) The JIT compiled functions, which are basically various helpers such as checking if two queens attack each other,
   computing the energy, and running a MCMC step.
2) The solver class which runs the whole simulation and keeps track of useful info.
"""

import jax
import jax.numpy as jnp
from functools import partial
from .board import BoardState
import time
import numpy as np


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
def compute_delta_energy_jit(queen_idx, old_pos, new_pos, queens):
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
    return new_attacks - old_attacks


@jax.jit 
def compute_total_energy_jit(queens):
    """
    Compute total energy (number of attacking pairs) - JIT-compiled.

    Args:
        queens: all position of all queens
    """
    n = queens.shape[0]
    
    def check_pair(i, j):
        return jnp.where(i < j, check_attack_jit(queens[i], queens[j]).astype(jnp.int32), 0)
    
    # Create all pairs
    i_indices, j_indices = jnp.meshgrid(jnp.arange(n), jnp.arange(n), indexing='ij')
    pairs = jax.vmap(jax.vmap(check_pair))(i_indices, j_indices)
    
    return jnp.sum(pairs)


@partial(jax.jit, static_argnums=(1,))
def mcmc_step(state, N, key, beta):
    """
    Single MCMC step - fully JIT-compiled.

    Args:
        state: current state of the board, of the MH chain (so of the queens), and of beta
        N: edge of the 3d cube
        key: for random utils
        beta: temperature
    
    Returns: (new_queens, new_board, new_energy, delta_J, accepted, key)
    """
    queens, board, energy = state
    
    # Select random queen
    key, subkey = jax.random.split(key)
    queen_idx = jax.random.randint(subkey, (), 0, N**2)
    old_pos = queens[queen_idx]
    
    # Select random cell
    key, subkey = jax.random.split(key)
    new_pos = jax.random.randint(subkey, (3,), 0, N)
    
    # Check if occupied
    is_occupied = board[new_pos[0], new_pos[1], new_pos[2]]
    
    # Compute delta energy
    delta_J = compute_delta_energy_jit(queen_idx, old_pos, new_pos, queens)
    delta_J = jnp.where(is_occupied, 0.0, delta_J.astype(jnp.float32))
    
    # Acceptance probability
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey)
    accept = (delta_J <= 0) | (u < jnp.exp(-beta * delta_J))
    accept = accept & ~is_occupied
    
    # Update state conditionally
    new_queens = jnp.where(accept, queens.at[queen_idx].set(new_pos), queens)
    
    # Update board
    new_board = jnp.where(
        accept,
        board.at[old_pos[0], old_pos[1], old_pos[2]].set(False).at[new_pos[0], new_pos[1], new_pos[2]].set(True),
        board
    )
    
    new_energy = jnp.where(accept, energy + delta_J, energy)
    
    return (new_queens, new_board, new_energy), delta_J, accept, key


class MCMCSolver:
    """
    MCMC solver for 3D N² Queens problem using Metropolis-Hastings.
    
    Features:
    - Multiple proposal types (move, swap, greedy)
    - Simulated annealing with adaptive/geometric cooling
    - Parallel tempering option
    - JIT-compiled for performance
    """
    def __init__(self, board_state):
        self.state = board_state
        self.N = board_state.N
    
    def run_improved(self, key, num_steps, initial_beta, final_beta, cooling, proposal_mix):
        """
        Improved MCMC with multiple proposals and better cooling.
        
        Args:
            key: JAX random key
            num_steps: Number of MCMC steps
            initial_beta: Starting inverse temperature
            final_beta: Final inverse temperature
            cooling: 'linear', 'geometric', or 'adaptive'
            proposal_mix: (move_prob, swap_prob, greedy_prob) - must sum to 1
                - move: Move single queen to random empty cell
                - swap: Swap positions of two queens
                - greedy: Move queen with most conflicts
        """
        N = self.N
        queens = self.state.queens.astype(jnp.int32)
        board = self.state.board
        energy = jnp.array(float(self.state.energy), dtype=jnp.float32)

        # Track best state
        best_queens = queens
        best_energy = float(energy)
        
        energy_history = [float(energy)]
        accepted_count = 0
        
        # Proposal probabilities
        move_prob, swap_prob, greedy_prob = proposal_mix
        
        print("="*60)
        print(f"3D Queens MCMC Solver (N={N}) - Improved")
        print("="*60)
        print(f"Board: {N}×{N}×{N} = {N**3} cells, Queens: {N**2}")
        print(f"Steps: {num_steps}, Cooling: {cooling}")
        print(f"β: {initial_beta} --> {final_beta}")
        print(f"Proposals: move={move_prob:.0%}, swap={swap_prob:.0%}, greedy={greedy_prob:.0%}")
        print(f"Initial energy: {energy}")
        print("="*60)
        
        start_time = time.time()
        beta = initial_beta
        
        # For geometric cooling
        if cooling == 'geometric':
            cooling_rate = (final_beta / initial_beta) ** (1.0 / num_steps)
        
        # Adaptive parameters
        window_size = 1000
        recent_accepts = []
        target_accept_rate = 0.3
        
        print_interval = max(1, num_steps // 10)
        
        for step in range(num_steps):
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
            
            # Select proposal type
            # key, subkey = jax.random.split(key)
            # r = float(jax.random.uniform(subkey))
            
            state_tuple = (queens, board, energy)
            beta_jnp = jnp.array(beta, dtype=jnp.float32)
            
            # if r < move_prob:
            #     # Standard move proposal
            #     (queens, board, energy), delta_J, accepted, key = mcmc_step(
            #         state_tuple, N, key, beta_jnp
            #     )
            # elif r < move_prob + swap_prob:
            #     # Swap proposal
            #     (queens, board, energy), delta_J, accepted, key = mcmc_step_swap(
            #         state_tuple, N, key, beta_jnp
            #     )
            # else:
            #     # Greedy move proposal
            #     (queens, board, energy), delta_J, accepted, key = mcmc_step_greedy_move(
            #         state_tuple, N, key, beta_jnp
            #     )

            (queens, board, energy), delta_J, accepted, key = mcmc_step(
                state_tuple, N, key, beta_jnp
            )

            accepted_count += int(accepted)
            recent_accepts.append(int(accepted))
            
            # Track best
            current_energy = float(energy)
            if current_energy < best_energy:
                best_energy = current_energy
                best_queens = queens
            
            # Track energy
            if num_steps <= 10000 or step % 100 == 0:
                energy_history.append(current_energy)
            
            # Print progress
            if step % print_interval == 0 or step == num_steps - 1:
                elapsed = time.time() - start_time
                rate = (step + 1) / elapsed if elapsed > 0 else 0
                print(f"Step {step+1:>7}/{num_steps}: "
                      f"E={current_energy:>5.0f}, "
                      f"Best={best_energy:>4.0f}, "
                      f"Acc={accepted_count/(step+1):>5.1%}, "
                      f"β={beta:>6.2f}, "
                      f"{rate:>6.0f}/s")
            
            if current_energy == 0:
                print(f"\nSOLUTION FOUND at step {step+1}!")
                break
        
        # Reconstruct best state
        self.state.queens = best_queens
        self.state.board = jnp.zeros((N, N, N), dtype=bool)
        self.state.board = self.state.board.at[best_queens[:,0], best_queens[:,1], best_queens[:,2]].set(True)
        self.state.energy = jnp.array(best_energy)
        
        elapsed = time.time() - start_time
        
        print("="*60)
        print(f"Time: {elapsed:.1f}s, Rate: {(step+1)/elapsed:.0f}/s")
        print(f"Best energy: {best_energy}, Solution: {'Yes' if best_energy == 0 else 'No'}")
        print("="*60)
        
        return self.state, np.array(energy_history), accepted_count / (step + 1)
    

        
    def run(self, key, num_steps, initial_beta=0.1, final_beta=10.0, adaptive=True):
        """
        Run MCMC with adaptive simulated annealing.
        
        Optimized with JAX JIT compilation.
        """
        # Convert state to JIT-friendly format
        queens = self.state.queens.astype(jnp.int32)
        board = self.state.board
        energy = jnp.array(float(self.state.energy), dtype=jnp.float32)
        
        # Track best state
        best_queens = queens
        best_energy = float(energy)
        
        energy_history = [float(energy)]
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
        print(f"Steps: {num_steps}")
        print(f"Initial β: {initial_beta}, Final β: {final_beta}")
        print(f"Initial energy: {energy}")
        if adaptive:
            print(f"Adaptive: Reheat after {reheat_threshold}, Restart after {restart_threshold}")
        print("="*60)
        print("Compiling JIT functions (first run may be slow)...")
        
        start_time = time.time()
        beta = initial_beta
        
        # Print progress every 10% of steps
        print_interval = max(1, num_steps // 1000)
        
        for step in range(num_steps):
            # Standard annealing schedule
            scheduled_beta = initial_beta + (final_beta - initial_beta) * (step / num_steps)
            
            if adaptive:
                current_energy = float(energy)
                if current_energy >= last_energy:
                    stuck_count += 1
                else:
                    stuck_count = 0
                    last_energy = current_energy
                
                # Reheat if stuck
                if stuck_count > reheat_threshold and stuck_count % reheat_threshold == 0:
                    beta = max(initial_beta, beta * 0.5)
                
                # Restart from best if stuck too long
                if stuck_count > restart_threshold:
                    queens = best_queens
                    board = jnp.zeros((self.N, self.N, self.N), dtype=bool)
                    board = board.at[queens[:,0], queens[:,1], queens[:,2]].set(True)
                    energy = jnp.array(best_energy, dtype=jnp.float32)
                    beta = initial_beta
                    stuck_count = 0
                
                beta = beta + 0.1 * (scheduled_beta - beta)
            else:
                beta = scheduled_beta
            
            # JIT-compiled MCMC step
            state_tuple = (queens, board, energy)
            (queens, board, energy), delta_J, accepted, key = mcmc_step(
                state_tuple, self.N, key, jnp.array(beta, dtype=jnp.float32)
            )
            
            accepted_count += int(accepted)
            
            # Track best
            current_energy = float(energy)
            if current_energy < best_energy:
                best_energy = current_energy
                best_queens = queens
            
            # Track energy (sample every 100 steps for large runs)
            if num_steps <= 10000 or step % 100 == 0:
                energy_history.append(current_energy)
            
            # Print progress
            if step % print_interval == 0 or step == num_steps - 1:
                elapsed = time.time() - start_time
                rate = (step + 1) / elapsed if elapsed > 0 else 0
                print(f"Step {step+1:>6}/{num_steps}: "
                      f"Energy={current_energy:>6.1f}, "
                      f"Best={best_energy:>4.0f}, "
                      f"Accept={accepted_count/(step+1):>5.1%}, "
                      f"β={beta:>5.3f}, "
                      f"Rate={rate:>6.0f}/s, "
                      f"Time={elapsed:>5.1f}s")
            
            # Check for solution
            if current_energy == 0:
                print(f"\n SOLUTION FOUND at step {step+1}!")
                break
        
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
        print("="*60)
        
        return self.state, np.array(energy_history), final_accept_rate
