"""
Beta Hyperparameter Optimization Script for 3D N² Queens Problem
No Simulated Annealing - Testing constant beta values

This script performs systematic hyperparameter optimization of constant beta values
on an 11×11×11 board without simulated annealing.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from src.board import BoardState
from src.solver import MCMCSolver
from src.visualize import visualize_solution, visualize_latin_square
import math

# =============================================================================
# GLOBAL CONFIGURATION CONSTANTS
# =============================================================================

# Board Configuration
BOARD_SIZE = 11

# Optimization Configuration
BETA_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]  # Log-spaced beta values 0.01, 0.05, 0.1, 0.5, 1.0, 5.0,10.0, 50.0
NUM_RUNS_PER_BETA = 3
BASE_SEED = 42

# MCMC Configuration
STEPS = 2_000_000
METHOD = 'basic'
COMPLEXITY = 'iter'
ENERGY_REGROUND_INTERVAL = 100000000
ENERGY_TREATMENT = 'linear'

# Simulated Annealing (disabled for constant beta)
SIMULATED_ANNEALING = False
COOLING = 'linear'  # Ignored when SA is disabled


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def pad_history(history, target_length):
    """Pad history with its last value to match target length."""
    current_length = len(history)
    if current_length < target_length:
        padding = np.full(target_length - current_length, history[-1])
        return np.concatenate([history, padding])
    return history


def count_endangered_queens(solution):
    """
    Calculates the number of queens that are under attack by at least one other queen.
    Unlike 'energy' (which counts pairs), this counts specific queens.
    """
    queens = np.array(solution.queens)
    num_queens = len(queens)
    endangered_count = 0

    for i in range(num_queens):
        q1 = queens[i]
        is_endangered = False
        
        for j in range(num_queens):
            if i == j: 
                continue
            
            q2 = queens[j]
            
            # Calculate absolute differences
            delta = np.abs(q1 - q2)
            d_i, d_j, d_k = delta[0], delta[1], delta[2]
            
            # 1. Rook-type: share at least two coordinates
            matches = np.sum(q1 == q2)
            if matches >= 2:
                is_endangered = True
                break
            
            # 2. Planar diagonals
            if q1[2] == q2[2] and d_i == d_j:  # xy-plane
                is_endangered = True
                break
            if q1[1] == q2[1] and d_i == d_k:  # xz-plane
                is_endangered = True
                break
            if q1[0] == q2[0] and d_j == d_k:  # yz-plane
                is_endangered = True
                break

            # 3. Space diagonal
            if d_i == d_j == d_k:
                is_endangered = True
                break
        
        if is_endangered:
            endangered_count += 1
            
    return endangered_count


def run_solver(beta_value, seed, size):
    """
    Run the solver with constant beta (no simulated annealing).
    
    Args:
        beta_value: Constant beta value to use
        seed: Random seed
        size: Board size N
    
    Returns:
        solution: Final board state
        energy_history: Array of energy values over iterations
        metric: Acceptance rate
    """
    key = jax.random.PRNGKey(seed)
    board = BoardState(key, size)
    solver = MCMCSolver(board)
    
    if METHOD == 'basic':
        solution, energy_history, metric = solver.run(
            key, 
            num_steps=STEPS,
            initial_beta=beta_value,
            final_beta=beta_value,  # Same as initial for constant beta
            cooling=COOLING,
            simulated_annealing=SIMULATED_ANNEALING,
            complexity=COMPLEXITY,
            energy_reground_interval=ENERGY_REGROUND_INTERVAL,
            name_energy_treatment=ENERGY_TREATMENT
        )
    elif METHOD == 'improved':
        solution, energy_history, metric = solver.run_improved(
            key,
            num_steps=STEPS,
            initial_beta=beta_value,
            final_beta=beta_value,  # Same as initial for constant beta
            cooling=COOLING,
            simulated_annealing=SIMULATED_ANNEALING,
            complexity=COMPLEXITY,
            energy_reground_interval=ENERGY_REGROUND_INTERVAL,
            name_energy_treatment=ENERGY_TREATMENT
        )
    else:
        raise ValueError(f"Unknown method: {METHOD}")
        
    return solution, energy_history, metric


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_beta_comparison(all_histories, beta_values, filename='beta_comparison_N11.png'):
    """
    Plot averaged energy curves for all beta values.
    
    Args:
        all_histories: Dict mapping beta_idx to list of 5 energy histories
        beta_values: List of beta values tested
        filename: Output filename
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use a colormap for distinct colors
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(beta_values)))
    
    for beta_idx, beta in enumerate(beta_values):
        histories = all_histories[beta_idx]
        
        # Pad all histories to same length
        max_len = max(len(h) for h in histories)
        padded = [pad_history(h, max_len) for h in histories]
        
        # Compute mean
        mean_energy = np.mean(padded, axis=0)
        
        # Plot averaged curve
        ax.plot(mean_energy, color=colors[beta_idx], linewidth=2.5, 
                label=f'β = {beta}', alpha=0.9)
    
    ax.set_xlabel('Iteration', fontsize=13, fontweight='bold')
    ax.set_ylabel('Energy', fontsize=13, fontweight='bold')
    ax.set_title(f'Beta Hyperparameter Comparison: Averaged Energy Curves\n' +
                 f'Board: {BOARD_SIZE}×{BOARD_SIZE}×{BOARD_SIZE} | ' +
                 f'Steps: {STEPS:,} | Runs per β: {NUM_RUNS_PER_BETA}',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # Add metadata box
    metadata_text = '\n'.join([
        f'Board: {BOARD_SIZE}³ = {BOARD_SIZE**3:,} cells',
        f'Queens: {BOARD_SIZE**2}',
        f'Steps: {STEPS:,}',
        f'Method: {METHOD}',
        f'Complexity: {COMPLEXITY}',
        f'Runs per β: {NUM_RUNS_PER_BETA}'
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, metadata_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved beta comparison plot: {filename}")


def plot_best_energy_vs_beta(beta_values, min_energies, avg_energies, 
                              filename='optimal_beta_N11.png'):
    """
    Plot minimum final energy vs beta value.
    
    Args:
        beta_values: List of beta values tested
        min_energies: List of minimum energies (best of 5 runs) for each beta
        avg_energies: List of average energies for each beta
        filename: Output filename
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Find optimal beta
    optimal_idx = np.argmin(min_energies)
    optimal_beta = beta_values[optimal_idx]
    optimal_energy = min_energies[optimal_idx]
    
    # Plot with log scale on x-axis if beta values span multiple orders of magnitude
    if max(beta_values) / min(beta_values) > 100:
        ax.set_xscale('log')
    
    # Plot line with markers
    ax.plot(beta_values, min_energies, 'o-', color='darkblue', linewidth=2.5, 
            markersize=10, label='Minimum Energy (Best of 5 runs)', zorder=3)
    ax.plot(beta_values, avg_energies, 's--', color='steelblue', linewidth=2, 
            markersize=8, alpha=0.7, label='Average Energy (Mean of 5 runs)', zorder=2)
    
    # Highlight optimal point
    ax.plot(optimal_beta, optimal_energy, 'r*', markersize=20, 
            label=f'Optimal β = {optimal_beta}', zorder=4)
    
    # Add annotation for optimal point
    ax.annotate(f'Optimal β = {optimal_beta}\nEnergy = {optimal_energy:.0f}',
                xy=(optimal_beta, optimal_energy),
                xytext=(20, 20), textcoords='offset points',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=2))
    
    ax.set_xlabel('Beta (β)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Final Energy', fontsize=13, fontweight='bold')
    ax.set_title(f'Optimal Beta Selection: Final Energy vs β\n' +
                 f'Board: {BOARD_SIZE}×{BOARD_SIZE}×{BOARD_SIZE} | ' +
                 f'Steps: {STEPS:,} | Runs per β: {NUM_RUNS_PER_BETA}',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Add metadata box
    metadata_text = '\n'.join([
        f'Board Size: {BOARD_SIZE}×{BOARD_SIZE}×{BOARD_SIZE}',
        f'Steps: {STEPS:,}',
        f'Method: {METHOD}',
        f'No Simulated Annealing',
        f'Runs per β: {NUM_RUNS_PER_BETA}',
        f'',
        f'Optimal β: {optimal_beta}',
        f'Best Energy: {optimal_energy:.0f}'
    ])
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.7)
    ax.text(0.02, 0.98, metadata_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved optimal beta plot: {filename}")


# =============================================================================
# MAIN OPTIMIZATION ROUTINE
# =============================================================================

def main():
    """Main optimization routine."""
    print("=" * 80)
    print("BETA HYPERPARAMETER OPTIMIZATION")
    print("3D N² Queens Problem - No Simulated Annealing")
    print("=" * 80)
    print(f"Board: {BOARD_SIZE}×{BOARD_SIZE}×{BOARD_SIZE} | Steps: {STEPS:,} | Runs per β: {NUM_RUNS_PER_BETA}")
    print(f"Beta values to test: {BETA_VALUES}")
    print(f"Method: {METHOD} | Complexity: {COMPLEXITY}")
    print("=" * 80 + "\n")
    
    # Check solvability
    gcd = math.gcd(BOARD_SIZE, 210)
    solvable = (gcd == 1)
    if solvable:
        print(f"✓ Board size N={BOARD_SIZE} is SOLVABLE (gcd(N,210)=1)")
        print("  A zero-energy solution exists (Klarner's theorem)\n")
    else:
        print(f"⚠ Board size N={BOARD_SIZE} may be UNSOLVABLE (gcd(N,210)={gcd})")
        print("  No guarantee of zero-energy solution\n")
    
    # Data structures
    all_histories = {}  # beta_idx -> list of 5 histories
    final_energies = {}  # beta_idx -> list of 5 final energies
    best_solutions = {}  # beta_idx -> best solution for that beta
    acceptance_rates = {}  # beta_idx -> list of 5 acceptance rates
    
    # Outer loop: iterate over beta values
    for beta_idx, beta in enumerate(BETA_VALUES):
        print(f"\n{'='*60}")
        print(f"Testing Beta = {beta}")
        print(f"{'='*60}")
        
        histories_for_beta = []
        energies_for_beta = []
        solutions_for_beta = []
        rates_for_beta = []
        
        # Inner loop: 5 runs per beta
        for run_idx in range(NUM_RUNS_PER_BETA):
            seed = BASE_SEED + run_idx
            print(f"  Run {run_idx+1}/{NUM_RUNS_PER_BETA} (Seed {seed})...", end=' ', flush=True)
            
            solution, energy_history, acceptance_rate = run_solver(beta, seed, BOARD_SIZE)
            
            final_energy = float(solution.energy)
            histories_for_beta.append(energy_history)
            energies_for_beta.append(final_energy)
            solutions_for_beta.append(solution)
            rates_for_beta.append(acceptance_rate)
            
            print(f"Energy = {final_energy:.0f}, Acceptance = {acceptance_rate:.2%}")
        
        # Store results for this beta
        all_histories[beta_idx] = histories_for_beta
        final_energies[beta_idx] = energies_for_beta
        acceptance_rates[beta_idx] = rates_for_beta
        
        # Find best solution for this beta
        best_idx = np.argmin(energies_for_beta)
        best_solutions[beta_idx] = solutions_for_beta[best_idx]
        
        # Summary for this beta
        avg_energy = np.mean(energies_for_beta)
        min_energy = np.min(energies_for_beta)
        max_energy = np.max(energies_for_beta)
        avg_acceptance = np.mean(rates_for_beta)
        
        print(f"\n  Beta {beta} Summary:")
        print(f"    Min Energy: {min_energy:.0f}")
        print(f"    Avg Energy: {avg_energy:.2f}")
        print(f"    Max Energy: {max_energy:.0f}")
        print(f"    Avg Acceptance: {avg_acceptance:.2%}")
    
    # =============================================================================
    # RESULTS SUMMARY
    # =============================================================================
    
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)
    print(f"{'Beta':<10} {'Min Energy':<15} {'Avg Energy':<15} {'Max Energy':<15} {'Avg Accept':<12}")
    print("-"*80)
    
    min_energies_list = []
    avg_energies_list = []
    
    for beta_idx, beta in enumerate(BETA_VALUES):
        energies = final_energies[beta_idx]
        rates = acceptance_rates[beta_idx]
        min_e = np.min(energies)
        avg_e = np.mean(energies)
        max_e = np.max(energies)
        avg_r = np.mean(rates)
        
        min_energies_list.append(min_e)
        avg_energies_list.append(avg_e)
        
        print(f"{beta:<10.2f} {min_e:<15.0f} {avg_e:<15.2f} {max_e:<15.0f} {avg_r:<12.2%}")
    
    # Find globally optimal beta
    global_optimal_idx = np.argmin(min_energies_list)
    global_optimal_beta = BETA_VALUES[global_optimal_idx]
    global_optimal_energy = min_energies_list[global_optimal_idx]
    
    print("-"*80)
    print(f"\n{'★'*3} OPTIMAL BETA: {global_optimal_beta} (Energy: {global_optimal_energy:.0f}) {'★'*3}\n")
    
    # =============================================================================
    # GENERATE PLOTS
    # =============================================================================
    
    print("="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    # Plot 1: Averaged energy curves for all betas
    plot_beta_comparison(all_histories, BETA_VALUES)
    
    # Plot 2: Best energy vs beta
    plot_best_energy_vs_beta(BETA_VALUES, min_energies_list, avg_energies_list)
    
    # =============================================================================
    # SAVE BEST SOLUTION VISUALIZATIONS
    # =============================================================================
    
    print("\n" + "="*80)
    print("SAVING BEST SOLUTION VISUALIZATIONS")
    print("="*80)
    
    # Get globally best solution
    best_beta_idx = global_optimal_idx
    best_solution = best_solutions[best_beta_idx]
    best_beta_value = BETA_VALUES[best_beta_idx]
    
    endangered = count_endangered_queens(best_solution)
    
    # 3D visualization
    sol_filename = f"best_solution_beta{best_beta_value}_N{BOARD_SIZE}.png"
    visualize_solution(best_solution, endangered, sol_filename)
    print(f"Saved 3D visualization: {sol_filename}")
    
    # Latin square visualization
    latin_filename = f"best_latin_square_beta{best_beta_value}_N{BOARD_SIZE}.png"
    visualize_latin_square(best_solution, endangered, latin_filename)
    print(f"Saved Latin square: {latin_filename}")
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Best configuration: β = {best_beta_value}")
    print(f"Best energy: {global_optimal_energy:.0f}")
    print(f"Endangered queens: {endangered}")
    if global_optimal_energy == 0:
        print("✓ ZERO-ENERGY SOLUTION FOUND!")
    print("="*80)


if __name__ == "__main__":
    main()

