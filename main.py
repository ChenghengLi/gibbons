import argparse
import jax
import math
import yaml
import numpy as np
from src.board import BoardState
from src.solver import MCMCSolver
from src.visualize import visualize_solution, plot_energy_history, plot_averaged_energy_history, visualize_latin_square
import matplotlib.pyplot as plt


def check_solvability(N):
    """
    Check if the 3D N² Queens problem is theoretically solvable.
    
    Based on Klarner's theorem (1967):
    - If gcd(N, 210) = 1 a solution exists (sufficient condition, conjectured necessary)
    - We write unsolvable if gcd(N, 210) != 1 (even though it might be solvable)
    - 210 = 2 × 3 × 5 × 7
    - This means N must not be divisible by 2, 3, 5, or 7
    
    Known solvable N (up to 20): 1, 11, 13, 17, 19, ...
    Known unsolvable N: 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, ...
    """
    gcd = math.gcd(N, 210)
    solvable = (gcd == 1)
    
    # Factor 210 to show which primes divide N
    factors = []
    if N % 2 == 0: factors.append(2)
    if N % 3 == 0: factors.append(3)
    if N % 5 == 0: factors.append(5)
    if N % 7 == 0: factors.append(7)
    
    return {
        'N': N,
        'gcd_210': gcd,
        'solvable': solvable,
        'factors': factors,
        'queens': N**2,
        'cells': N**3,
        'density': N**2 / N**3
    }


def print_solvability_info(info):
    """Print solvability information."""
    print("="*60)
    print("PROBLEM SOLVABILITY CHECK")
    print("="*60)
    print(f"Board size N = {info['N']}")
    print(f"Queens: {info['queens']}, Cells: {info['cells']}, Density: {info['density']:.1%}")
    print(f"gcd(N, 210) = gcd({info['N']}, 2×3×5×7) = {info['gcd_210']}")
    
    if info['solvable']:
        print(f"\nSOLVABLE: N={info['N']} has gcd(N,210)=1")
        print("   A zero-energy solution EXISTS (Klarner's theorem)")
    else:
        print(f"N={info['N']} is divisible by {info['factors']}")
        print("   No guarantees that the zero-energy solution exists!")
        print("   The algorithm will find the MINIMUM energy configuration.")
    print("="*60 + "\n")


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_solver(config, seed, size):
    """Run the solver with parameters from config and specific seed and size."""
    steps = config['steps']
    method = config['method']
    cooling = config['cooling']
    beta_min = config['beta_min']
    beta_max = config['beta_max']
    simulated_annealing = config.get('simulated_annealing')
    complexity = config.get('complexity')
    energy_reground_interval = config.get('energy_reground_interval', 0)
    energy_treatment = config.get('energy_treatment')
    
    # Create fresh state and solver for each run to avoid state leakage
    key = jax.random.PRNGKey(seed)
    board = BoardState(key, size)
    solver = MCMCSolver(board)
    
    print(f"Running 3D {size}×{size}×{size} Queens with {steps} steps (Seed: {seed})...")
    print(f"Method: {method}, Cooling: {cooling}, SA: {simulated_annealing}, Complexity: {complexity}")
    
    if method == 'basic':
        solution, energy_history, metric = solver.run(
            key, 
            num_steps=steps,
            initial_beta=beta_min,
            final_beta=beta_max,
            cooling=cooling,
            simulated_annealing=simulated_annealing,
            complexity=complexity,
            energy_reground_interval=energy_reground_interval,
            name_energy_treatment=energy_treatment
        )
    elif method == 'improved':
        solution, energy_history, metric = solver.run_improved(
            key,
            num_steps=steps,
            initial_beta=beta_min,
            final_beta=beta_max,
            cooling=cooling,
            simulated_annealing=simulated_annealing,
            complexity=complexity,
            energy_reground_interval=energy_reground_interval,
            name_energy_treatment=energy_treatment
        )
    else:
        raise ValueError(f"Unknown method: {method}")
        
    return solution, energy_history, metric


def pad_history(history, target_length):
    """Pad history with its last value to match target length."""
    current_length = len(history)
    if current_length < target_length:
        padding = np.full(target_length - current_length, history[-1])
        return np.concatenate([history, padding])
    return history


def main():
    parser = argparse.ArgumentParser(description='3D N² Queens MCMC Solver')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Config file '{args.config}' not found.")
        return
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # Handle 'sizes' list or fallback to single 'size'
    sizes = config.get('sizes')
    if sizes is None:
        if 'size' in config:
            sizes = [config['size']]
        else:
            print("Error: No 'sizes' or 'size' specified in config.")
            return
            
    if not isinstance(sizes, list):
        sizes = [sizes]

    mode = config.get('mode')
    if mode not in ('single', 'multiple'):
        raise ValueError(f"Error: mode = {mode}")

    show_plots = config.get('show', False)
    
    for size in sizes:
        print("\n" + "="*80)
        print(f"STARTING RUNS FOR BOARD SIZE N={size}")
        print("="*80 + "\n")
        
        # Check solvability first
        solvability = check_solvability(size)
        print_solvability_info(solvability)
        
        if mode == 'single':
            base_seed = config.get('seed', 42)
            solution, energy_history, metric = run_solver(config, base_seed, size)

            endangered = count_endangered_queens(solution)
            print(f"Endangered Queens: {endangered}")
            
            print("\nResults:")
            print(f"Final energy: {solution.energy}")
            print(f"Acceptance rate: {metric:.2%}")

            if float(solution.energy) == 0:
                print("Valid solution found!")
            else:
                print("No zero-energy solution found.")
                
            # Visualize
            sol_file = f"solution_N{size}_seed{base_seed}.png"
            visualize_solution(solution, sol_file)
            print(f"\n3D visualization saved as {sol_file}")
            
            latin_file = f"latin_square_N{size}_seed{base_seed}.png"
            visualize_latin_square(solution, latin_file)
            print(f"Latin square visualization saved as {latin_file}")
            
            energy_file = f"energy_history_N{size}_seed{base_seed}.png"
            plot_energy_history(energy_history, energy_file)
            print(f"Energy history saved as {energy_file}")
            
            if show_plots:
                print("\nDisplaying plots (close windows to exit)...")
                visualize_solution(solution, endangered)
                visualize_latin_square(solution, endangered)
                plot_energy_history(energy_history)
                plt.show()
                
        elif mode == 'multiple':
            num_runs = config.get('num_runs', 5)
            print(f"Executing {num_runs} runs in '{mode}' mode for N={size}...")
            
            # Simple incremental seed generation
            base_seed = config.get('seed', 42)
            seeds = [base_seed + i for i in range(num_runs)]
            
            histories = []
            final_energies = []
            solutions = []
            
            for i, seed in enumerate(seeds):
                print(f"\n--- Run {i+1}/{num_runs} (N={size}) ---")
                solution, energy_history, metric = run_solver(config, seed, size)
                
                histories.append(energy_history)
                final_energies.append(float(solution.energy))
                solutions.append(solution)
                
                print(f"Run {i+1} Result: Energy={solution.energy}")
            
            # Determine max length for padding
            max_len = max(len(h) for h in histories)
            padded_histories = [pad_history(h, max_len) for h in histories]
            
            # Statistics
            avg_energy = np.mean(final_energies)
            min_energy = np.min(final_energies)
            success_rate = sum(e == 0 for e in final_energies) / num_runs
            
            print("\n" + "="*60)
            print(f"MULTIPLE RUNS SUMMARY (N={size}, {num_runs} runs)")
            print("="*60)
            print(f"Average Final Energy: {avg_energy:.2f}")
            print(f"Minimum Final Energy: {min_energy}")
            print(f"Success Rate: {success_rate:.1%}")
            
            # Prepare metadata for plot
            simulated_annealing = config.get('simulated_annealing', True)
            beta_range = f"{config['beta_min']} -> {config['beta_max']}" if simulated_annealing else f"{config['beta_min']} (Constant)"
            
            metadata = {
                'beta_range': beta_range,
                'steps': config['steps'],
                'cooling_method': config['cooling'] if simulated_annealing else 'None',
                'board_size': size,
                'final_energy': avg_energy
            }
            
            # Visualize Averaged Energy
            avg_energy_file = f"averaged_energy_history_N{size}.png"
            plot_averaged_energy_history(padded_histories, avg_energy_file, metadata=metadata)
            print(f"\nAveraged energy history saved as {avg_energy_file}")
            
            # Visualize Best Solution
            best_idx = np.argmin(final_energies)
            best_solution = solutions[best_idx]
            endangered = count_endangered_queens(best_solution)
            print(f"Endangered Queens: {endangered}")
            best_sol_file = f"best_solution_N{size}.png"
            visualize_solution(best_solution, best_sol_file)
            print(f"Best solution visualization saved as {best_sol_file}")
            
            best_latin_file = f"best_latin_square_N{size}.png"
            visualize_latin_square(best_solution, best_latin_file)
            print(f"Best Latin square visualization saved as {best_latin_file}")
            
            if show_plots:
                print("\nDisplaying plots (close windows to exit)...")
                plot_averaged_energy_history(padded_histories, metadata=metadata)
                visualize_solution(best_solution)
                visualize_latin_square(best_solution)
                plt.show()

    else:
        print(f"Error: Unknown mode '{mode}' in config. Use 'single' or 'multiple'.")


def count_endangered_queens(solution):
    """
    Calculates the number of queens that are under attack by at least one other queen.
    Unlike 'energy' (which counts pairs), this counts specific queens.
    """
    # Convert JAX array to numpy for standard iteration
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
            
            # --- Attack Logic (Same as BoardState) ---
            # Calculate absolute differences
            delta = np.abs(q1 - q2)
            d_i, d_j, d_k = delta[0], delta[1], delta[2]
            
            # 1. Rook-type: share at least two coordinates
            # Sum of boolean matches (i==i) + (j==j) + (k==k)
            matches = np.sum(q1 == q2)
            if matches >= 2:
                is_endangered = True
                break
            
            # 2. Planar diagonals (share 1 coord, diff of others is equal)
            # xy-plane (k is same)
            if q1[2] == q2[2] and d_i == d_j:
                is_endangered = True
                break
            # xz-plane (j is same)
            if q1[1] == q2[1] and d_i == d_k:
                is_endangered = True
                break
            # yz-plane (i is same)
            if q1[0] == q2[0] and d_j == d_k:
                is_endangered = True
                break

            # 3. Space diagonal (all 3 diffs equal)
            if d_i == d_j == d_k:
                is_endangered = True
                break
        
        if is_endangered:
            endangered_count += 1
            
    return endangered_count

if __name__ == "__main__":
    main()