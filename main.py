"""
Main entry point for 3D N² Queens MCMC Solver (Refactored).

This script provides a modular, config-driven interface to run the solver
with either full or reduced state space.

Usage:
    python main.py --config config.yaml
    python main.py --config config.yaml --state-space reduced
    python main.py --size 5 --steps 100000 --state-space full
"""

import argparse
import sys
from typing import List, Tuple, Optional
import numpy as np

import jax

from src.config import Config
from src.board import FullBoardState, ReducedBoardState
from src.solver import FullStateSolver, ReducedStateSolver, create_solver
from src.utils import check_solvability
from src.visualize import (
    visualize_solution,
    visualize_latin_square,
    plot_energy_history,
    plot_averaged_energy_history,
    save_results,
    save_multiple_results,
    save_run_results,
    save_competition_format,
)


# =============================================================================
# Runner Classes
# =============================================================================

class SolverRunner:
    """
    Orchestrates solver execution based on configuration.
    """
    
    def __init__(self, config: Config):
        """
        Initialize runner with configuration.
        
        Args:
            config: Configuration instance
        """
        self.config = config
    
    def run_single(self, size: int, seed: int) -> Tuple[any, np.ndarray, float]:
        """
        Run a single solver execution.
        
        Args:
            size: Board dimension N
            seed: Random seed
            
        Returns:
            Tuple of (board_state, energy_history, acceptance_rate)
        """
        key = jax.random.PRNGKey(seed)
        
        # Create board based on state space type
        if self.config.state_space == 'reduced':
            board = ReducedBoardState(key, size)
            solver = ReducedStateSolver(board, seed)
        else:
            board = FullBoardState(key, size)
            solver = FullStateSolver(board, seed)
        
        # Run solver
        result_board, energy_history, accept_rate = solver.run(
            num_steps=self.config.steps,
            initial_beta=self.config.beta_min,
            final_beta=self.config.beta_max,
            cooling=self.config.cooling,
            simulated_annealing=self.config.simulated_annealing,
            energy_treatment=self.config.energy_treatment,
            complexity=self.config.complexity,
            energy_reground_interval=self.config.energy_reground_interval,
            log_interval=self.config.log_interval,
            verbose=True
        )
        
        return result_board, energy_history, accept_rate
    
    def run_multiple(self, size: int) -> List[Tuple[any, np.ndarray, float]]:
        """
        Run multiple solver executions with different seeds.
        
        Args:
            size: Board dimension N
            
        Returns:
            List of (board_state, energy_history, acceptance_rate) tuples
        """
        results = []
        base_seed = self.config.seed
        
        for i in range(self.config.num_runs):
            seed = base_seed + i
            print(f"\n{'='*60}")
            print(f"Run {i+1}/{self.config.num_runs} (seed={seed})")
            print(f"{'='*60}")
            
            result = self.run_single(size, seed)
            results.append(result)
        
        return results
    
    def run(self) -> dict:
        """
        Execute solver based on configuration.
        
        Returns:
            Dictionary with results for each board size
        """
        all_results = {}
        
        for size in self.config.sizes:
            print(f"\n{'#'*60}")
            print(f"# Board Size N = {size}")
            print(f"{'#'*60}")
            
            # Print solvability info
            solvability = check_solvability(size)
            self._print_solvability(solvability)
            
            if self.config.mode == 'single':
                board, history, accept_rate = self.run_single(size, self.config.seed)
                all_results[size] = {
                    'board': board,
                    'history': history,
                    'accept_rate': accept_rate,
                    'final_energy': board.energy,
                }
            else:
                results = self.run_multiple(size)
                
                # Aggregate statistics
                final_energies = [r[0].energy for r in results]
                histories = [r[1] for r in results]
                accept_rates = [r[2] for r in results]
                
                # Find best
                best_idx = np.argmin(final_energies)
                
                all_results[size] = {
                    'boards': [r[0] for r in results],
                    'histories': histories,
                    'accept_rates': accept_rates,
                    'final_energies': final_energies,
                    'best_board': results[best_idx][0],
                    'best_energy': final_energies[best_idx],
                    'avg_energy': np.mean(final_energies),
                    'success_rate': sum(e == 0 for e in final_energies) / len(final_energies),
                }
                
                self._print_multiple_summary(size, all_results[size])
        
        return all_results
    
    def _print_solvability(self, info: dict) -> None:
        """Print solvability information."""
        print(f"\nSolvability Check for N={info['N']}:")
        print(f"  Queens: {info['queens']}, Cells: {info['cells']}")
        print(f"  Density: {info['density']:.1%}")
        print(f"  gcd(N, 210) = {info['gcd_210']}")
        
        if info['solvable']:
            print(f"  ✓ SOLVABLE: Zero-energy solution exists")
        else:
            print(f"  ✗ May be UNSOLVABLE: N divisible by {info['blocking_factors']}")
        print()
    
    def _print_multiple_summary(self, size: int, results: dict) -> None:
        """Print summary for multiple runs."""
        print(f"\n{'='*60}")
        print(f"Summary for N={size} ({len(results['final_energies'])} runs)")
        print(f"{'='*60}")
        print(f"Average final energy: {results['avg_energy']:.2f}")
        print(f"Best energy: {results['best_energy']:.0f}")
        print(f"Success rate: {results['success_rate']:.1%}")
        print(f"{'='*60}")


# =============================================================================
# CLI Interface
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='3D N² Queens MCMC Solver',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to YAML configuration file'
    )
    
    # Override options
    parser.add_argument(
        '--size', '-n',
        type=int,
        nargs='+',
        help='Board size(s) N (overrides config)'
    )
    
    parser.add_argument(
        '--steps', '-s',
        type=int,
        help='Number of MCMC steps (overrides config)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed (overrides config)'
    )
    
    parser.add_argument(
        '--state-space',
        type=str,
        choices=['full', 'reduced'],
        help='State space type (overrides config)'
    )
    
    parser.add_argument(
        '--complexity',
        type=str,
        choices=['hash', 'iter', 'endangered'],
        help='Energy computation: hash (O(1)), iter (O(N²) pairs), or endangered (O(N²) queens)'
    )
    
    parser.add_argument(
        '--log-interval',
        type=int,
        help='Log progress every N steps (0 = auto, default 10%% of steps)'
    )
    
    parser.add_argument(
        '--cooling',
        type=str,
        choices=['linear', 'geometric', 'adaptive'],
        help='Cooling schedule (overrides config)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['single', 'multiple'],
        help='Execution mode (overrides config)'
    )
    
    parser.add_argument(
        '--num-runs',
        type=int,
        help='Number of runs for multiple mode (overrides config)'
    )
    
    parser.add_argument(
        '--no-annealing',
        action='store_true',
        help='Disable simulated annealing'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save results (plots and data)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for saved results'
    )
    
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show plots interactively'
    )
    
    return parser.parse_args()


def load_config_with_overrides(args: argparse.Namespace) -> Config:
    """
    Load configuration from file and apply CLI overrides.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Configuration with overrides applied
    """
    try:
        config = Config.from_yaml(args.config)
    except FileNotFoundError:
        print(f"Warning: Config file '{args.config}' not found, using defaults")
        config = Config()
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Apply CLI overrides
    if args.size:
        config.sizes = args.size
    if args.steps:
        config.steps = args.steps
    if args.seed:
        config.seed = args.seed
    if args.state_space:
        config.state_space = args.state_space
    if args.complexity:
        config.complexity = args.complexity
    if args.log_interval is not None:
        config.log_interval = args.log_interval
    if args.cooling:
        config.cooling = args.cooling
    if args.mode:
        config.mode = args.mode
    if args.num_runs:
        config.num_runs = args.num_runs
    if args.no_annealing:
        config.simulated_annealing = False
    if args.save:
        config.save = True
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.show:
        config.show = True
    
    return config


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration
    config = load_config_with_overrides(args)
    
    # Validate
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    # Print configuration
    config.print_summary()
    
    # Run solver
    runner = SolverRunner(config)
    results = runner.run()
    
    # Final summary
    print(f"\n{'#'*60}")
    print("# Final Results")
    print(f"{'#'*60}")
    
    for size, result in results.items():
        if config.mode == 'single':
            energy = result['final_energy']
            status = "✓ SOLVED" if energy == 0 else f"Best energy: {energy:.0f}"
            print(f"N={size}: {status}")
        else:
            print(f"N={size}: Best={result['best_energy']:.0f}, "
                  f"Avg={result['avg_energy']:.2f}, "
                  f"Success={result['success_rate']:.1%}")
    
    # Always save results to timestamped folders
    print(f"\n{'#'*60}")
    print("# Saving Results")
    print(f"{'#'*60}")
    
    for size, result in results.items():
        metadata = {
            'board_size': size,
            'steps': config.steps,
            'seed': config.seed,
            'state_space': config.state_space,
            'cooling': config.cooling,
            'beta_min': config.beta_min,
            'beta_max': config.beta_max,
            'simulated_annealing': config.simulated_annealing,
            'energy_treatment': config.energy_treatment,
        }
        
        if config.mode == 'single':
            board = result['board']
            history = result['history']
            metadata['accept_rate'] = result['accept_rate']
            
            # Always save to timestamped folder with competition format
            saved = save_run_results(
                config.output_dir, board, history, metadata,
                save_plots=config.save, save_data=config.save
            )
            print(f"N={size}: Saved to {saved['run_folder']}/")
            print(f"  - solution.txt (competition format)")
            print(f"  - metadata.json")
            if config.save:
                print(f"  - solution.png, latin_square.png, energy_history.png")
            
            if config.show:
                visualize_solution(board, show=True, metadata=metadata)
                visualize_latin_square(board, show=True, metadata=metadata)
                plot_energy_history(history, show=True, metadata=metadata)
        else:
            boards = result['boards']
            histories = result['histories']
            accept_rates = result['accept_rates']
            metadata['num_runs'] = config.num_runs
            
            # Save each run to its own timestamped folder
            for i, (board, history, acc_rate) in enumerate(zip(boards, histories, accept_rates)):
                run_metadata = metadata.copy()
                run_metadata['seed'] = config.seed + i
                run_metadata['run_index'] = i + 1
                run_metadata['accept_rate'] = acc_rate
                
                saved = save_run_results(
                    config.output_dir, board, history, run_metadata,
                    save_plots=config.save, save_data=config.save
                )
                print(f"N={size} Run {i+1}: Saved to {saved['run_folder']}/")
            
            # Also save summary for multiple runs
            if config.save:
                results_tuples = [(b, h, r) for b, h, r in 
                                  zip(boards, histories, accept_rates)]
                saved = save_multiple_results(
                    config.output_dir, results_tuples, histories, metadata,
                    save_plots=True, save_data=True
                )
                print(f"N={size}: Summary saved to {config.output_dir}/")
            
            if config.show:
                plot_averaged_energy_history(histories, show=True, metadata=metadata)
                visualize_solution(result['best_board'], show=True, metadata=metadata)
    
    print()


if __name__ == "__main__":
    main()
