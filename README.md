# 3D N² Queens MCMC Solver

**EPFL - Markov Chains and Algorithmic Applications**

Solve the 3D N²-Queens problem using Markov Chain Monte Carlo (MCMC) with JAX acceleration. Place N² queens on an N×N×N cubic chessboard such that no two queens attack each other.

---

## Quick Start

```bash
# Install dependencies
conda env create -f environment.yml
conda activate markov-queens

# Run with config file
python main.py --config config.yaml

# Or with command line options
python main.py --size 5 --steps 100000 --state-space reduced
```

---

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Configuration](#configuration)
4. [State Spaces](#state-spaces)
5. [Project Structure](#project-structure)
6. [API Reference](#api-reference)

---

## Installation

### Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate markov-queens
```

### Using pip

```bash
pip install -r requirements.txt
```

### Dependencies
- Python 3.8+
- JAX
- NumPy
- PyYAML
- Matplotlib (optional, for visualization)

---

## Usage

### Command Line Interface

```bash
# Basic usage with config file
python main.py --config config.yaml

# Override config options via CLI
python main.py --size 5 --steps 100000
python main.py --size 11 --state-space reduced --cooling geometric
python main.py --mode multiple --num-runs 10

# Full example
python main.py \
    --size 5 6 7 \
    --steps 500000 \
    --state-space reduced \
    --cooling geometric \
    --seed 42
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config`, `-c` | Path to YAML config file | `config.yaml` |
| `--size`, `-n` | Board size(s) N (can specify multiple) | from config |
| `--steps`, `-s` | Number of MCMC steps | from config |
| `--seed` | Random seed | from config |
| `--state-space` | `full` or `reduced` | from config |
| `--cooling` | `linear`, `geometric`, or `adaptive` | from config |
| `--mode` | `single` or `multiple` | from config |
| `--num-runs` | Number of runs for multiple mode | from config |
| `--complexity` | `hash`, `iter`, or `endangered` | from config |
| `--log-interval` | Steps between progress logs (0=auto) | from config |
| `--no-annealing` | Disable simulated annealing | False |
| `--verbose`, `-v` | Verbose output | False |
| `--save` | Save results (plots and data) | False |
| `--output-dir`, `-o` | Output directory for saved results | `results` |
| `--show` | Show plots interactively | False |

---

## Configuration

Create a `config.yaml` file to configure the solver:

```yaml
# =============================================================================
# Board Configuration
# =============================================================================
sizes: [5, 6, 7]           # List of board sizes N to run

# =============================================================================
# Solver Configuration  
# =============================================================================
steps: 1000000             # Number of MCMC steps
seed: 42                   # Random seed for reproducibility

# State Space Options:
#   - full: General state space, C(N³, N²) configurations
#           Queens can be placed anywhere
#   - reduced: Restricted state space, N^(N²) configurations
#              One queen per (i,j) pair, only k varies
state_space: reduced

# Cooling Schedule Options:
#   - linear: β increases linearly from beta_min to beta_max
#   - geometric: β increases geometrically (exponential)
#   - adaptive: β adjusts based on acceptance rate
cooling: geometric

# Temperature Parameters
beta_min: 0.1              # Initial inverse temperature
beta_max: 25.0             # Final inverse temperature
simulated_annealing: true  # Enable/disable annealing (false = constant beta)

# Energy Treatment Options:
#   - linear: Use energy directly (E)
#   - quadratic: Use E²
#   - log: Use log(1 + E)
#   - log_quadratic: Use log(1 + E²)
energy_treatment: linear

# Complexity Options:
#   - hash: O(1) energy updates using line counting
#   - iter: O(N²) energy updates counting attacking pairs
#   - endangered: O(N²) energy updates counting endangered queens
complexity: hash

# Energy Regrounding (for numerical stability, hash only)
energy_reground_interval: 10000  # Steps between recalculations (0 = disable)

# Log Interval: how often to print progress (0 = auto, default 10% of steps)
log_interval: 100000

# =============================================================================
# Execution Configuration
# =============================================================================
# Mode Options:
#   - single: Run once with the specified seed
#   - multiple: Run num_runs times with different seeds
mode: single
num_runs: 10               # Number of runs if mode is 'multiple'

# =============================================================================
# Output Configuration
# =============================================================================
show: false                # Show plots interactively
save: false                # Save results (plots and data)
output_dir: results        # Directory for saved results
```

### Configuration Options Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `sizes` | list[int] | `[5]` | Board sizes to run |
| `steps` | int | `1000000` | MCMC iterations |
| `seed` | int | `42` | Random seed |
| `state_space` | str | `full` | `full` or `reduced` |
| `cooling` | str | `geometric` | `linear`, `geometric`, `adaptive` |
| `beta_min` | float | `0.1` | Initial β |
| `beta_max` | float | `25.0` | Final β |
| `simulated_annealing` | bool | `true` | Enable annealing |
| `energy_treatment` | str | `linear` | Energy function |
| `complexity` | str | `hash` | `hash`, `iter`, or `endangered` |
| `energy_reground_interval` | int | `0` | Regrounding interval |
| `log_interval` | int | `0` | Progress log interval (0=auto) |
| `mode` | str | `single` | `single` or `multiple` |
| `num_runs` | int | `1` | Runs for multiple mode |
| `show` | bool | `false` | Show plots |
| `save` | bool | `false` | Save results |
| `output_dir` | str | `results` | Output directory |

---

## State Spaces

### Full State Space (`state_space: full`)

- **Configuration**: Any N² positions on the N×N×N cube
- **State space size**: C(N³, N²) ≈ very large
- **Proposal**: Move random queen to random empty cell
- **Use when**: Exploring all possible configurations

### Reduced State Space (`state_space: reduced`)

- **Configuration**: One queen per (i,j) pair at height k(i,j)
- **State space size**: N^(N²) — much smaller
- **Proposal**: Change k-coordinate for random (i,j)
- **Guarantees**: No rook attacks in ij-direction
- **Use when**: Faster convergence, still finds solutions

**Recommendation**: Use `reduced` for most cases — it's faster and still correct.

---

## Energy Computation (Complexity)

### Hash (`complexity: hash`) - O(1) per step

- Uses **line counting** data structure
- Tracks queens on each of 13 line families (rooks, diagonals, space diagonals)
- Energy = sum of C(n,2) for each line with n queens
- **Exact energy**: Correctly counts attacking pairs (same as iter)
- **Fast**: Constant time per MCMC step regardless of N

### Iter (`complexity: iter`) - O(N²) per step

- Uses **iterative attack checking**
- Directly counts attacking pairs by checking all queen pairs
- **True energy**: Exact count of attacking pairs
- **Slower**: Linear in number of queens (N²) per step

### Endangered (`complexity: endangered`) - O(N²) per step

- Uses **iterative attack checking**
- Counts number of **endangered queens** (queens under attack by at least one other)
- **Alternative energy**: Minimizes number of queens at risk
- **Slower**: Linear in number of queens (N²) per step

### Which to use?

| Scenario | Recommendation |
|----------|----------------|
| Large N (≥10) | `hash` — much faster |
| Small N (<10) | `iter` — similar speed, exact energy |
| Accuracy needed | `iter` — true energy (attacking pairs) |
| Speed needed | `hash` — O(1) updates |
| Alternative metric | `endangered` — minimize queens at risk |

**Note**: All methods correctly identify solutions (energy = 0). Hash and iter now compute the **same exact energy** (attacking pairs). The metadata saves `final_energy_hash`, `final_energy_iter`, and `final_energy_endangered` for verification.

---

## Project Structure

```
Markov/
├── main.py              # Entry point with CLI
├── config.yaml          # Configuration file
├── src/
│   ├── __init__.py      # Package exports
│   ├── interfaces.py    # Abstract base classes
│   ├── board.py         # FullBoardState, ReducedBoardState
│   ├── solver.py        # FullStateSolver, ReducedStateSolver
│   ├── config.py        # Config dataclass
│   └── utils.py         # JIT utilities, energy functions
├── tests/
│   ├── __init__.py
│   └── test_refactored.py
├── environment.yml
├── requirements.txt
└── README.md
```

### Module Overview

| Module | Description |
|--------|-------------|
| `src/interfaces.py` | `BoardInterface`, `SolverInterface` abstract classes |
| `src/board.py` | Board state implementations |
| `src/solver.py` | MCMC solver implementations with JIT |
| `src/config.py` | Configuration management |
| `src/utils.py` | Attack checking, line indices, energy treatments |

---

## API Reference

### Using as a Library

```python
import jax
from src.board import FullBoardState, ReducedBoardState
from src.solver import FullStateSolver, ReducedStateSolver, create_solver
from src.config import Config

# Create board
key = jax.random.PRNGKey(42)
board = ReducedBoardState(key, N=5)

# Create solver
solver = ReducedStateSolver(board, seed=42)

# Run MCMC
result, history, accept_rate = solver.run(
    num_steps=100000,
    initial_beta=0.1,
    final_beta=25.0,
    cooling='geometric',
    simulated_annealing=True,
    energy_treatment='linear',
    complexity='hash',        # 'hash', 'iter', or 'endangered'
    log_interval=10000,       # Log every 10k steps
    verbose=True
)

print(f"Final energy: {result.energy}")
print(f"Acceptance rate: {accept_rate:.1%}")
```

### Board Classes

```python
# Full state space
board = FullBoardState(key, N)
board.get_queens()      # (N², 3) array of positions
board.get_board()       # (N, N, N) occupancy grid
board.energy            # Current energy
board.compute_energy()  # Recompute from scratch

# Reduced state space
board = ReducedBoardState(key, N)
board.get_k_config()    # (N, N) array of k-coordinates
board.get_queens()      # Derived (N², 3) array
```

### Solver Classes

```python
# Factory function (auto-selects solver type)
solver = create_solver(board, seed=42)

# Or explicitly
solver = FullStateSolver(board, seed=42)
solver = ReducedStateSolver(board, seed=42)

# Run
result, history, accept_rate = solver.run(
    num_steps=100000,
    initial_beta=0.1,
    final_beta=25.0,
    cooling='geometric',       # 'linear', 'geometric', 'adaptive'
    simulated_annealing=True,
    energy_treatment='linear', # 'linear', 'quadratic', 'log', 'log_quadratic'
    complexity='hash',         # 'hash', 'iter', or 'endangered'
    energy_reground_interval=10000,
    log_interval=10000,        # 0 = auto (10% of steps)
    verbose=True
)
```

---

## Solvability

Based on Klarner's theorem, a zero-energy solution exists iff `gcd(N, 210) = 1`.

| N | Solvable | Notes |
|---|----------|-------|
| 1 | ✅ | Trivial |
| 2-10 | ❌ | No solution exists |
| 11 | ✅ | First non-trivial solvable |
| 13, 17, 19, 23... | ✅ | gcd(N, 210) = 1 |

---

## Running Tests

```bash
python tests/test_refactored.py
```

---

## License

Educational project for EPFL Markov Chains course.
