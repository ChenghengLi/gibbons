import argparse
import jax
import math
import yaml
import numpy as np
from src.board import BoardState
from src.solver import MCMCSolver
from src.visualize import visualize_solution, plot_energy_history, plot_averaged_energy_history
import matplotlib.pyplot as plt


def check_solvability(N):
    """
    Check if the 3D N² Queens problem is theoretically solvable.
    
    Based on Klarner's theorem (1967):
    - A solution exists if gcd(N, 210) = 1
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
        print(f"\nUNSOLVABLE: N={info['N']} is divisible by {info['factors']}")
        print("   No zero-energy solution exists!")
        print("   The algorithm will find the MINIMUM energy configuration.")
    print("="*60 + "\n")


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_solver(config, seed):
    """Run the solver with parameters from config and specific seed."""
    size = config['size']
    steps = config['steps']
    method = config['method']
    cooling = config['cooling']
    beta_min = config['beta_min']
    beta_max = config['beta_max']
    simulated_annealing = config.get('simulated_annealing', True)
    
    key = jax.random.PRNGKey(seed)
    board = BoardState(key, size)
    solver = MCMCSolver(board)
    
    print(f"Running 3D {size}×{size}×{size} Queens with {steps} steps (Seed: {seed})...")
    print(f"Method: {method}, Cooling: {cooling}, SA: {simulated_annealing}")
    
    if method == 'basic':
        solution, energy_history, metric = solver.run(
            key, 
            num_steps=steps,
            initial_beta=beta_min,
            final_beta=beta_max,
            simulated_annealing=simulated_annealing
        )
    elif method == 'improved':
        solution, energy_history, metric = solver.run_improved(
            key,
            num_steps=steps,
            initial_beta=beta_min,
            final_beta=beta_max,
            cooling=cooling,
            proposal_mix=(0.5, 0.3, 0.2),  # move, swap, greedy
            simulated_annealing=simulated_annealing
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

    # Check solvability first
    solvability = check_solvability(config['size'])
    print_solvability_info(solvability)
    
    mode = config.get('mode', 'single')
    base_seed = config['seed']
    show_plots = config.get('show', False)
    
    if mode == 'single':
        solution, energy_history, metric = run_solver(config, base_seed)
        
        print("\nResults:")
        print(f"Final energy: {solution.energy}")
        print(f"Acceptance rate: {metric:.2%}")

        if float(solution.energy) == 0:
            print("Valid solution found!")
        else:
            print("No zero-energy solution found.")
            
        # Visualize
        sol_file = f"solution_{config['size']}x{config['size']}x{config['size']}.png"
        visualize_solution(solution, sol_file)
        print(f"\n3D visualization saved as {sol_file}")
        
        energy_file = 'energy_history.png'
        plot_energy_history(energy_history, energy_file)
        print(f"Energy history saved as {energy_file}")
        
        if show_plots:
            print("\nDisplaying plots (close windows to exit)...")
            visualize_solution(solution)
            plot_energy_history(energy_history)
            plt.show()
            
    elif mode == 'multiple':
        num_runs = config.get('num_runs', 5)
        print(f"Executing {num_runs} runs in '{mode}' mode...")
        
        histories = []
        final_energies = []
        solutions = []
        
        for i in range(num_runs):
            print(f"\n--- Run {i+1}/{num_runs} ---")
            seed = base_seed + i
            solution, energy_history, metric = run_solver(config, seed)
            
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
        print(f"MULTIPLE RUNS SUMMARY ({num_runs} runs)")
        print("="*60)
        print(f"Average Final Energy: {avg_energy:.2f}")
        print(f"Minimum Final Energy: {min_energy}")
        print(f"Success Rate: {success_rate:.1%}")
        
        # Visualize Averaged Energy
        avg_energy_file = 'averaged_energy_history.png'
        plot_averaged_energy_history(padded_histories, avg_energy_file)
        print(f"\nAveraged energy history saved as {avg_energy_file}")
        
        # Visualize Best Solution
        best_idx = np.argmin(final_energies)
        best_solution = solutions[best_idx]
        best_sol_file = f"best_solution_{config['size']}x{config['size']}x{config['size']}.png"
        visualize_solution(best_solution, best_sol_file)
        print(f"Best solution visualization saved as {best_sol_file}")
        
        if show_plots:
            print("\nDisplaying plots (close windows to exit)...")
            plot_averaged_energy_history(padded_histories)
            visualize_solution(best_solution)
            plt.show()

    else:
        print(f"Error: Unknown mode '{mode}' in config. Use 'single' or 'multiple'.")

if __name__ == "__main__":
    main()
