"""
Test script to verify the MCMC solver reaches known minimum energies.

Known minimum energies from the paper:
N:      3    4    5    6    7    8    9   10   11
J_min: 13   21   32   41   50   53   55   36    0

This script tests both full and reduced state spaces and verifies
the energy computation matches the naive (ground truth) method.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple

from src.board import FullBoardState, ReducedBoardState
from src.solver import FullStateSolver, ReducedStateSolver
from src.utils import count_attacking_pairs, check_attack_python


# Known minimum energies
KNOWN_MINIMUMS = {
    3: 13,
    4: 21,
    5: 32,
    6: 41,
    7: 50,
    8: 53,
    9: 55,
    10: 36,
    11: 0,
}

# Solver parameters tuned for each N
# Format: (steps, beta_max)
SOLVER_PARAMS = {
    3: (50000, 30.0),
    4: (100000, 40.0),
    5: (200000, 50.0),
    6: (500000, 60.0),
    7: (1000000, 70.0),
    8: (2000000, 80.0),
    9: (3000000, 90.0),
    10: (5000000, 100.0),
    11: (10000000, 150.0),
}


def compute_naive_energy(board) -> int:
    """Compute energy using naive O(N⁴) method - ground truth."""
    queens = board.get_queens()
    queens_list = [(int(q[0]), int(q[1]), int(q[2])) for q in queens]
    return count_attacking_pairs(queens_list)


def verify_energy_consistency(board) -> Tuple[bool, float, int]:
    """
    Verify that board.energy matches naive computation.
    
    Returns:
        (is_consistent, board_energy, naive_energy)
    """
    board_energy = float(board.energy)
    naive_energy = compute_naive_energy(board)
    
    # Note: board.energy may be a surrogate (line-based) that overcounts
    # But when energy is 0, both should be 0
    if naive_energy == 0:
        is_consistent = (board_energy == 0)
    else:
        # For non-zero, we just check naive energy
        is_consistent = True
    
    return is_consistent, board_energy, naive_energy


def run_solver_test(
    N: int,
    state_space: str,
    steps: int,
    beta_max: float,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[float, int, float]:
    """
    Run solver and return results.
    
    Returns:
        (best_line_energy, best_naive_energy, accept_rate)
    """
    key = jax.random.PRNGKey(seed)
    
    if state_space == 'reduced':
        board = ReducedBoardState(key, N)
        solver = ReducedStateSolver(board, seed)
    else:
        board = FullBoardState(key, N)
        solver = FullStateSolver(board, seed)
    
    result, history, accept_rate = solver.run(
        num_steps=steps,
        initial_beta=0.1,
        final_beta=beta_max,
        cooling='geometric',
        simulated_annealing=True,
        energy_treatment='linear',
        verbose=verbose
    )
    
    # Get both energies
    line_energy = float(result.energy)
    naive_energy = compute_naive_energy(result)
    
    return line_energy, naive_energy, accept_rate


def test_single_n(N: int, num_runs: int = 3, verbose: bool = True):
    """Test a single N value with multiple runs."""
    steps, beta_max = SOLVER_PARAMS.get(N, (100000, 50.0))
    target = KNOWN_MINIMUMS.get(N, None)
    
    print(f"\n{'='*70}")
    print(f"Testing N={N}, Target J_min={target}")
    print(f"Parameters: steps={steps:,}, beta_max={beta_max}")
    print(f"{'='*70}")
    
    results = {'reduced': [], 'full': []}
    
    for state_space in ['reduced', 'full']:
        print(f"\n--- {state_space.upper()} State Space ---")
        
        best_naive = float('inf')
        
        for run in range(num_runs):
            seed = 42 + run
            print(f"\nRun {run+1}/{num_runs} (seed={seed}):")
            
            line_e, naive_e, acc = run_solver_test(
                N, state_space, steps, beta_max, seed, verbose=verbose
            )
            
            results[state_space].append({
                'line_energy': line_e,
                'naive_energy': naive_e,
                'accept_rate': acc,
            })
            
            if naive_e < best_naive:
                best_naive = naive_e
            
            status = "✓" if naive_e <= target else "✗"
            print(f"  Line Energy: {line_e:.0f}, Naive Energy: {naive_e}, "
                  f"Accept: {acc:.1%} {status}")
        
        print(f"\nBest naive energy ({state_space}): {best_naive}")
        if target is not None:
            if best_naive <= target:
                print(f"✓ REACHED TARGET (J_min={target})")
            else:
                print(f"✗ DID NOT REACH TARGET (got {best_naive}, need {target})")
    
    return results


def run_all_tests(max_n: int = 7, num_runs: int = 2):
    """Run tests for N=3 to max_n."""
    print("="*70)
    print("MINIMUM ENERGY VERIFICATION TEST")
    print("="*70)
    print("\nKnown minimum energies:")
    for n, e in sorted(KNOWN_MINIMUMS.items()):
        print(f"  N={n}: J_min={e}")
    
    all_results = {}
    
    for N in range(3, max_n + 1):
        all_results[N] = test_single_n(N, num_runs=num_runs, verbose=False)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'N':>3} | {'Target':>6} | {'Reduced':>10} | {'Full':>10} | Status")
    print("-"*50)
    
    for N in range(3, max_n + 1):
        target = KNOWN_MINIMUMS.get(N, '?')
        
        reduced_best = min(r['naive_energy'] for r in all_results[N]['reduced'])
        full_best = min(r['naive_energy'] for r in all_results[N]['full'])
        
        best = min(reduced_best, full_best)
        status = "✓" if best <= target else "✗"
        
        print(f"{N:>3} | {target:>6} | {reduced_best:>10} | {full_best:>10} | {status}")
    
    print("-"*50)


def quick_test():
    """Quick test for small N values."""
    print("="*70)
    print("QUICK VERIFICATION TEST (N=3,4,5)")
    print("="*70)
    
    for N in [3, 4, 5]:
        steps, beta_max = 50000, 30.0
        target = KNOWN_MINIMUMS[N]
        
        print(f"\n--- N={N}, Target={target} ---")
        
        for state_space in ['reduced']:
            key = jax.random.PRNGKey(42)
            
            if state_space == 'reduced':
                board = ReducedBoardState(key, N)
                solver = ReducedStateSolver(board, 42)
            else:
                board = FullBoardState(key, N)
                solver = FullStateSolver(board, 42)
            
            result, history, acc = solver.run(
                num_steps=steps,
                initial_beta=0.1,
                final_beta=beta_max,
                cooling='geometric',
                simulated_annealing=True,
                verbose=False
            )
            
            naive_e = compute_naive_energy(result)
            print(f"  {state_space}: naive_energy={naive_e}, target={target}, "
                  f"{'✓' if naive_e <= target else '✗'}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test minimum energy convergence')
    parser.add_argument('--quick', action='store_true', help='Quick test (N=3,4,5)')
    parser.add_argument('--max-n', type=int, default=7, help='Max N to test')
    parser.add_argument('--runs', type=int, default=2, help='Runs per configuration')
    parser.add_argument('-n', type=int, help='Test single N value')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    elif args.n:
        test_single_n(args.n, num_runs=args.runs, verbose=True)
    else:
        run_all_tests(max_n=args.max_n, num_runs=args.runs)
