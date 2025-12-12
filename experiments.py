"""
Experiments for MCMC Mini-Project: 3D N² Queens Problem

This script runs experiments to answer the project questions:
1. Energy vs time curves (averaged over multiple runs)
2. Effect of simulated annealing cooling schedules
3. Minimal energy as function of N
4. Identify N values with significantly lower minimal energy
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from src.board import BoardState
from src.solver import MCMCSolver
import time
import os


def run_single_experiment(N, num_steps, method='improved', cooling='geometric', 
                          seed=42, beta_min=0.1, beta_max=100.0):
    """Run a single MCMC experiment and return results."""
    key = jax.random.PRNGKey(seed)
    board = BoardState(key, N)
    solver = MCMCSolver(board)
    
    start = time.time()
    
    if method == 'basic':
        solution, energy_history, acc = solver.run(
            key, num_steps, initial_beta=beta_min, final_beta=beta_max, adaptive=True
        )
    elif method == 'improved':
        solution, energy_history, acc = solver.run_improved(
            key, num_steps, initial_beta=beta_min, final_beta=beta_max,
            cooling=cooling, proposal_mix=(0.5, 0.3, 0.2)
        )
    elif method == 'parallel':
        solution, energy_history, swaps = solver.run_parallel_tempering(
            key, num_steps, num_replicas=8, beta_min=beta_min, beta_max=beta_max
        )
        acc = swaps
    
    elapsed = time.time() - start
    final_energy = float(solution.energy)
    
    return {
        'N': N,
        'method': method,
        'cooling': cooling,
        'steps': num_steps,
        'final_energy': final_energy,
        'energy_history': energy_history,
        'time': elapsed,
        'seed': seed
    }


def experiment_1_energy_vs_time(N=4, num_runs=5, num_steps=50000):
    """
    Task 1: Energy vs time for fixed N, averaged over multiple runs.
    """
    print(f"\n{'='*60}")
    print(f"Experiment 1: Energy vs Time (N={N}, {num_runs} runs)")
    print(f"{'='*60}")
    
    all_histories = []
    
    for run in range(num_runs):
        print(f"\nRun {run+1}/{num_runs}...")
        result = run_single_experiment(N, num_steps, method='improved', seed=run*100)
        all_histories.append(result['energy_history'])
        print(f"  Final energy: {result['final_energy']}")
    
    # Align histories to same length
    min_len = min(len(h) for h in all_histories)
    aligned = np.array([h[:min_len] for h in all_histories])
    
    # Compute mean and std
    mean_energy = np.mean(aligned, axis=0)
    std_energy = np.std(aligned, axis=0)
    
    # Plot
    plt.figure(figsize=(10, 6))
    steps = np.arange(len(mean_energy))
    plt.plot(steps, mean_energy, 'b-', label='Mean energy')
    plt.fill_between(steps, mean_energy - std_energy, mean_energy + std_energy, 
                     alpha=0.3, label='±1 std')
    plt.xlabel('MCMC Step')
    plt.ylabel('Energy (attacking pairs)')
    plt.title(f'Energy vs Time for N={N} (averaged over {num_runs} runs)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'exp1_energy_vs_time_N{N}.png', dpi=150)
    plt.close()
    
    print(f"\nSaved: exp1_energy_vs_time_N{N}.png")
    return mean_energy, std_energy


def experiment_2_simulated_annealing(N=4, num_runs=3, num_steps=50000):
    """
    Task 2: Compare different cooling schedules.
    """
    print(f"\n{'='*60}")
    print(f"Experiment 2: Simulated Annealing Comparison (N={N})")
    print(f"{'='*60}")
    
    cooling_schedules = ['linear', 'geometric', 'adaptive']
    results = {c: [] for c in cooling_schedules}
    
    for cooling in cooling_schedules:
        print(f"\nCooling: {cooling}")
        for run in range(num_runs):
            result = run_single_experiment(N, num_steps, method='improved', 
                                          cooling=cooling, seed=run*100)
            results[cooling].append(result)
            print(f"  Run {run+1}: final energy = {result['final_energy']}")
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Energy histories
    plt.subplot(1, 2, 1)
    colors = {'linear': 'blue', 'geometric': 'green', 'adaptive': 'red'}
    for cooling in cooling_schedules:
        histories = [r['energy_history'] for r in results[cooling]]
        min_len = min(len(h) for h in histories)
        aligned = np.array([h[:min_len] for h in histories])
        mean_energy = np.mean(aligned, axis=0)
        plt.plot(mean_energy, color=colors[cooling], label=cooling)
    
    plt.xlabel('Step')
    plt.ylabel('Energy')
    plt.title('Energy vs Step by Cooling Schedule')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Final energies
    plt.subplot(1, 2, 2)
    final_energies = {c: [r['final_energy'] for r in results[c]] for c in cooling_schedules}
    positions = np.arange(len(cooling_schedules))
    means = [np.mean(final_energies[c]) for c in cooling_schedules]
    stds = [np.std(final_energies[c]) for c in cooling_schedules]
    
    plt.bar(positions, means, yerr=stds, capsize=5, color=[colors[c] for c in cooling_schedules])
    plt.xticks(positions, cooling_schedules)
    plt.ylabel('Final Energy')
    plt.title('Final Energy by Cooling Schedule')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'exp2_annealing_comparison_N{N}.png', dpi=150)
    plt.close()
    
    print(f"\nSaved: exp2_annealing_comparison_N{N}.png")
    return results


def experiment_3_minimal_energy_vs_N(N_values=range(2, 12), num_steps=100000, num_runs=3):
    """
    Task 3: Minimal energy as function of N.
    """
    print(f"\n{'='*60}")
    print(f"Experiment 3: Minimal Energy vs N")
    print(f"{'='*60}")
    
    results = []
    
    for N in N_values:
        print(f"\nN = {N}:")
        best_energy = float('inf')
        energies = []
        
        for run in range(num_runs):
            # Adjust steps based on N
            steps = min(num_steps, max(10000, num_steps // (N // 3 + 1)))
            result = run_single_experiment(N, steps, method='improved', seed=run*100)
            energies.append(result['final_energy'])
            if result['final_energy'] < best_energy:
                best_energy = result['final_energy']
            print(f"  Run {run+1}: energy = {result['final_energy']}")
        
        results.append({
            'N': N,
            'best_energy': best_energy,
            'mean_energy': np.mean(energies),
            'std_energy': np.std(energies),
            'queens': N**2,
            'cells': N**3,
            'density': N**2 / N**3
        })
        print(f"  Best: {best_energy}, Mean: {np.mean(energies):.1f}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    N_vals = [r['N'] for r in results]
    best_energies = [r['best_energy'] for r in results]
    mean_energies = [r['mean_energy'] for r in results]
    std_energies = [r['std_energy'] for r in results]
    
    # Plot 1: Minimal energy vs N
    axes[0].errorbar(N_vals, mean_energies, yerr=std_energies, fmt='o-', capsize=5)
    axes[0].scatter(N_vals, best_energies, color='red', marker='*', s=100, label='Best found', zorder=5)
    axes[0].set_xlabel('N (board size)')
    axes[0].set_ylabel('Energy (attacking pairs)')
    axes[0].set_title('Minimal Energy vs Board Size N')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Normalized energy (energy / N²)
    normalized = [r['best_energy'] / r['queens'] for r in results]
    axes[1].plot(N_vals, normalized, 'o-')
    axes[1].set_xlabel('N (board size)')
    axes[1].set_ylabel('Energy / N² (per queen)')
    axes[1].set_title('Normalized Energy vs Board Size')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('exp3_minimal_energy_vs_N.png', dpi=150)
    plt.close()
    
    # Print table
    print("\n" + "="*70)
    print(f"{'N':>3} | {'Queens':>6} | {'Cells':>6} | {'Density':>7} | {'Best E':>7} | {'Mean E':>7}")
    print("-"*70)
    for r in results:
        print(f"{r['N']:>3} | {r['queens']:>6} | {r['cells']:>6} | {r['density']:>7.1%} | {r['best_energy']:>7.0f} | {r['mean_energy']:>7.1f}")
    print("="*70)
    
    print(f"\nSaved: exp3_minimal_energy_vs_N.png")
    return results


def experiment_4_identify_special_N(N_values=range(2, 16), num_steps=50000, num_runs=2):
    """
    Task 4: Identify N values with significantly lower minimal energy.
    
    Theory predicts solutions exist when gcd(N, 210) = 1.
    210 = 2 × 3 × 5 × 7
    """
    print(f"\n{'='*60}")
    print(f"Experiment 4: Identify Special N Values")
    print(f"{'='*60}")
    
    import math
    
    results = []
    
    for N in N_values:
        gcd_210 = math.gcd(N, 210)
        theoretically_solvable = (gcd_210 == 1)
        
        print(f"\nN = {N} (gcd(N,210)={gcd_210}, solvable={theoretically_solvable}):")
        
        best_energy = float('inf')
        for run in range(num_runs):
            result = run_single_experiment(N, num_steps, method='improved', seed=run*100)
            if result['final_energy'] < best_energy:
                best_energy = result['final_energy']
        
        results.append({
            'N': N,
            'gcd_210': gcd_210,
            'theoretically_solvable': theoretically_solvable,
            'best_energy': best_energy,
            'normalized_energy': best_energy / (N**2)
        })
        print(f"  Best energy: {best_energy}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    N_vals = [r['N'] for r in results]
    energies = [r['best_energy'] for r in results]
    solvable = [r['theoretically_solvable'] for r in results]
    
    colors = ['green' if s else 'red' for s in solvable]
    bars = ax.bar(N_vals, energies, color=colors, alpha=0.7)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='gcd(N,210)=1 (solvable)'),
        Patch(facecolor='red', alpha=0.7, label='gcd(N,210)≠1 (unsolvable)')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    ax.set_xlabel('N (board size)')
    ax.set_ylabel('Best Energy Found')
    ax.set_title('Minimal Energy by N (colored by theoretical solvability)')
    ax.set_xticks(N_vals)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('exp4_special_N_values.png', dpi=150)
    plt.close()
    
    # Print summary
    print("\n" + "="*70)
    print(f"{'N':>3} | {'gcd(N,210)':>10} | {'Solvable?':>10} | {'Best E':>7} | {'E/N²':>7}")
    print("-"*70)
    for r in results:
        print(f"{r['N']:>3} | {r['gcd_210']:>10} | {'Yes' if r['theoretically_solvable'] else 'No':>10} | {r['best_energy']:>7.0f} | {r['normalized_energy']:>7.3f}")
    print("="*70)
    
    print(f"\nSaved: exp4_special_N_values.png")
    return results


def run_all_experiments():
    """Run all experiments for the mini-project."""
    print("\n" + "="*60)
    print("MCMC Mini-Project: 3D N² Queens Problem")
    print("Running all experiments...")
    print("="*60)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Experiment 1: Energy vs time
    exp1_results = experiment_1_energy_vs_time(N=4, num_runs=5, num_steps=30000)
    
    # Experiment 2: Simulated annealing comparison
    exp2_results = experiment_2_simulated_annealing(N=4, num_runs=3, num_steps=30000)
    
    # Experiment 3: Minimal energy vs N
    exp3_results = experiment_3_minimal_energy_vs_N(N_values=range(2, 10), num_steps=50000, num_runs=2)
    
    # Experiment 4: Identify special N values
    exp4_results = experiment_4_identify_special_N(N_values=range(2, 13), num_steps=30000, num_runs=2)
    
    print("\n" + "="*60)
    print("All experiments completed!")
    print("Generated plots:")
    print("  - exp1_energy_vs_time_N4.png")
    print("  - exp2_annealing_comparison_N4.png")
    print("  - exp3_minimal_energy_vs_N.png")
    print("  - exp4_special_N_values.png")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        exp = sys.argv[1]
        if exp == '1':
            experiment_1_energy_vs_time()
        elif exp == '2':
            experiment_2_simulated_annealing()
        elif exp == '3':
            experiment_3_minimal_energy_vs_N()
        elif exp == '4':
            experiment_4_identify_special_N()
        else:
            print("Usage: python experiments.py [1|2|3|4|all]")
    else:
        run_all_experiments()
