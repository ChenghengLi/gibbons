"""
Test convergence to known minimum energies for N=3 to N=11.
"""

import sys
sys.path.insert(0, '/Users/chenghengli/Desktop/Markov')

import jax
import time
from src.board import ReducedBoardState, FullBoardState
from src.solver import ReducedStateSolver, FullStateSolver

# Known minimum energies
KNOWN_MINIMUMS = {3: 13, 4: 21, 5: 32, 6: 41, 7: 50, 8: 53, 9: 55, 10: 36, 11: 0}

# Test parameters - use 500k steps for quick test, increase beta_max
NUM_STEPS = 500000
BETA_MAX_VALUES = [100, 500, 1000]
MAX_SEEDS = 3

results = {}

print('=' * 80)
print('MINIMUM ENERGY CONVERGENCE TEST')
print('=' * 80)
print(f'Steps: {NUM_STEPS:,}')
print(f'Beta max values to try: {BETA_MAX_VALUES}')
print(f'Max seeds per config: {MAX_SEEDS}')
print('=' * 80)

for N in range(3, 12):
    target = KNOWN_MINIMUMS[N]
    results[N] = {'target': target, 'reduced': None, 'full': None}
    
    print(f'\n### N={N} (Target: {target}) ###')
    
    # Test Reduced State Space
    print('  Reduced State Space:')
    best_reduced = float('inf')
    best_reduced_config = None
    
    for beta_max in BETA_MAX_VALUES:
        if best_reduced <= target:
            break
        for seed in range(MAX_SEEDS):
            if best_reduced <= target:
                break
            
            key = jax.random.PRNGKey(seed)
            board = ReducedBoardState(key, N)
            solver = ReducedStateSolver(board, seed)
            
            start = time.time()
            result, _, _ = solver.run(
                num_steps=NUM_STEPS,
                initial_beta=0.1,
                final_beta=beta_max,
                cooling='geometric',
                simulated_annealing=True,
                complexity='iter',
                verbose=False
            )
            elapsed = time.time() - start
            
            energy = int(result.energy)
            if energy < best_reduced:
                best_reduced = energy
                best_reduced_config = (beta_max, seed)
            
            status = 'REACHED' if energy <= target else ''
            print(f'    beta_max={beta_max}, seed={seed}: E={energy} ({elapsed:.1f}s) {status}')
            
            if energy <= target:
                break
    
    results[N]['reduced'] = best_reduced
    results[N]['reduced_config'] = best_reduced_config
    
    # Test Full State Space (skip for large N to save time)
    if N <= 8:
        print('  Full State Space:')
        best_full = float('inf')
        best_full_config = None
        
        for beta_max in BETA_MAX_VALUES:
            if best_full <= target:
                break
            for seed in range(MAX_SEEDS):
                if best_full <= target:
                    break
                
                key = jax.random.PRNGKey(seed)
                board = FullBoardState(key, N)
                solver = FullStateSolver(board, seed)
                
                start = time.time()
                result, _, _ = solver.run(
                    num_steps=NUM_STEPS,
                    initial_beta=0.1,
                    final_beta=beta_max,
                    cooling='geometric',
                    simulated_annealing=True,
                    complexity='iter',
                    verbose=False
                )
                elapsed = time.time() - start
                
                energy = int(result.energy)
                if energy < best_full:
                    best_full = energy
                    best_full_config = (beta_max, seed)
                
                status = 'REACHED' if energy <= target else ''
                print(f'    beta_max={beta_max}, seed={seed}: E={energy} ({elapsed:.1f}s) {status}')
                
                if energy <= target:
                    break
        
        results[N]['full'] = best_full
        results[N]['full_config'] = best_full_config
    else:
        print('  Full State Space: Skipped (N too large)')
        results[N]['full'] = 'skipped'

# Final Report
print('\n')
print('=' * 80)
print('FINAL REPORT')
print('=' * 80)
print('  N | Target | Reduced |    Full | Status')
print('-' * 50)

for N in range(3, 12):
    target = KNOWN_MINIMUMS[N]
    reduced = results[N]['reduced']
    full = results[N]['full']
    
    reduced_str = str(reduced) if reduced != float('inf') else 'N/A'
    full_str = str(full) if full not in [float('inf'), 'skipped'] else (full if full == 'skipped' else 'N/A')
    
    reduced_ok = reduced <= target if reduced != float('inf') else False
    full_ok = full <= target if full not in [float('inf'), 'skipped'] else False
    
    if reduced_ok or full_ok:
        status = 'REACHED'
    else:
        status = 'NOT REACHED'
    
    print(f'{N:>3} | {target:>6} | {reduced_str:>7} | {full_str:>7} | {status}')

print('=' * 80)
