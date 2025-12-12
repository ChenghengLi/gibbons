import argparse
import jax
import math
from src.board import BoardState
from src.solver import MCMCSolver
from src.visualize import visualize_solution, plot_energy_history
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
        print(f"\n✅ SOLVABLE: N={info['N']} has gcd(N,210)=1")
        print("   A zero-energy solution EXISTS (Klarner's theorem)")
    else:
        print(f"\n❌ UNSOLVABLE: N={info['N']} is divisible by {info['factors']}")
        print("   No zero-energy solution exists!")
        print("   The algorithm will find the MINIMUM energy configuration.")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='3D N² Queens MCMC Solver')
    parser.add_argument('--size', type=int, default=3, help='Board size (N)')
    parser.add_argument('--steps', type=int, default=10000, help='Number of MCMC steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--method', type=str, default='basic', 
                        choices=['basic', 'improved', 'parallel'],
                        help='MCMC method: basic, improved, or parallel (tempering)')
    parser.add_argument('--cooling', type=str, default='geometric',
                        choices=['linear', 'geometric', 'adaptive'],
                        help='Cooling schedule for simulated annealing')
    parser.add_argument('--beta-min', type=float, default=0.1, help='Initial/min beta')
    parser.add_argument('--beta-max', type=float, default=50.0, help='Final/max beta')
    parser.add_argument('--replicas', type=int, default=8, help='Number of replicas for parallel tempering')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    args = parser.parse_args()
    
    # Check solvability first
    solvability = check_solvability(args.size)
    print_solvability_info(solvability)
    
    print(f"Running 3D {args.size}×{args.size}×{args.size} Queens with {args.steps} steps...")
    print(f"Method: {args.method}, Cooling: {args.cooling}")
    
    # Initialize board and solver
    key = jax.random.PRNGKey(args.seed)
    board = BoardState(key, args.size)
    solver = MCMCSolver(board)
    
    # Run selected MCMC method
    if args.method == 'basic':
        solution, energy_history, acceptance_rate = solver.run(
            key, 
            num_steps=args.steps,
            initial_beta=args.beta_min,
            final_beta=args.beta_max
        )
    elif args.method == 'improved':
        solution, energy_history, acceptance_rate = solver.run_improved(
            key,
            num_steps=args.steps,
            initial_beta=args.beta_min,
            final_beta=args.beta_max,
            cooling=args.cooling,
            proposal_mix=(0.5, 0.3, 0.2)  # move, swap, greedy
        )
    elif args.method == 'parallel':
        solution, energy_history, swap_count = solver.run_parallel_tempering(
            key,
            num_steps=args.steps,
            num_replicas=args.replicas,
            beta_min=args.beta_min,
            beta_max=args.beta_max
        )
        acceptance_rate = None  # Different metric for parallel tempering
    
    print("\nResults:")
    print(f"Final energy: {solution.energy}")
    if acceptance_rate is not None:
        print(f"Acceptance rate: {acceptance_rate:.2%}")
    else:
        print(f"Replica swaps: {swap_count}")
    
    if float(solution.energy) == 0:
        print("✅ Valid solution found!")
    else:
        print("❌ No zero-energy solution found.")
    
    # Always visualize the best state found
    sol_file = f'solution_{args.size}x{args.size}x{args.size}.png'
    visualize_solution(solution, sol_file)
    print(f"\n📊 3D visualization saved as {sol_file}")
    
    # Plot energy history
    energy_file = 'energy_history.png'
    plot_energy_history(energy_history, energy_file)
    print(f"📈 Energy history saved as {energy_file}")
    
    # Show plots interactively if requested
    if args.show:
        print("\n🖼️  Displaying plots (close windows to exit)...")
        visualize_solution(solution)
        plot_energy_history(energy_history)
        plt.show()


if __name__ == "__main__":
    main()
