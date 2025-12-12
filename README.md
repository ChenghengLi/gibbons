# 3D N² Queens MCMC Solver

**EPFL - Markov Chains and Algorithmic Applications - Fall 2025-2026**

This project solves the 3D N²-Queens problem using Markov Chain Monte Carlo (MCMC) methods with JAX acceleration. The goal is to place N² queens on an N×N×N cubic chessboard such that no two queens attack each other.

---

## Table of Contents
1. [Problem Description](#problem-description)
2. [Theoretical Background](#theoretical-background)
3. [Solvability Conditions](#solvability-conditions)
4. [MCMC Methods](#mcmc-methods)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Project Structure](#project-structure)
8. [Experiments](#experiments)
9. [Results](#results)
10. [Troubleshooting](#troubleshooting)

---

## Problem Description

On a 3D N×N×N chessboard, place N² queens such that no two queens attack each other.

### Board and Configuration
- **Board**: B_N = {0, 1, ..., N-1}³ (a discrete cube with N³ cells)
- **Configuration**: A set s of N² distinct queen positions
- **State Space**: |S| = C(N³, N²) possible configurations

### Attack Relations (Definition 2.2)

Two queens at positions (i,j,k) and (i',j',k') attack each other if:

| Attack Type | Condition | Description |
|-------------|-----------|-------------|
| **Rook-type** | Share ≥2 coordinates | Same row/column in any plane |
| **Planar diagonal (xy)** | k=k' and \|i-i'\|=\|j-j'\|≠0 | Diagonal in xy-plane |
| **Planar diagonal (xz)** | j=j' and \|i-i'\|=\|k-k'\|≠0 | Diagonal in xz-plane |
| **Planar diagonal (yz)** | i=i' and \|j-j'\|=\|k-k'\|≠0 | Diagonal in yz-plane |
| **Space diagonal** | \|i-i'\|=\|j-j'\|=\|k-k'\|≠0 | 3D diagonal |

### Energy Function (Definition 2.5)

```
J(s) = number of attacking pairs in configuration s
```

A valid solution has **J(s) = 0**.

---

## Theoretical Background

### Solvability Conditions

Based on **Klarner's theorem (1967)**, a zero-energy solution exists if and only if:

```
gcd(N, 210) = 1
```

where 210 = 2 × 3 × 5 × 7.

| N | gcd(N, 210) | Solvable? | Minimum Energy |
|---|-------------|-----------|----------------|
| 1 | 1 | ✅ Yes | 0 |
| 2 | 2 | ❌ No | 6 |
| 3 | 3 | ❌ No | 13 |
| 4 | 2 | ❌ No | ~21 |
| 5 | 5 | ❌ No | ~35 |
| 6 | 6 | ❌ No | ~50 |
| 7 | 7 | ❌ No | ~70 |
| 11 | 1 | ✅ Yes | 0 |
| 13 | 1 | ✅ Yes | 0 |

**First solvable N > 1**: N = 11, 13, 17, 19, 23, ...

### MCMC Approach

The algorithm uses the **Metropolis-Hastings** algorithm:

1. **Proposal**: Move a queen to a new random position
2. **Acceptance**: Accept with probability α(s→s') = min(1, exp(-β × ΔJ))
3. **Simulated Annealing**: Gradually increase β to focus on low-energy states

---

## MCMC Methods

This implementation provides three MCMC methods:

### 1. Basic Method (`--method basic`)
- Standard Metropolis-Hastings with single-queen moves
- Linear simulated annealing schedule
- Adaptive reheating and restart

### 2. Improved Method (`--method improved`)
- **Multiple proposal types**:
  - Move (50%): Move single queen to random cell
  - Swap (30%): Swap positions of two queens
  - Greedy (20%): Move queen with most conflicts
- **Cooling schedules**: linear, geometric, adaptive
- Better exploration of state space

### 3. Parallel Tempering (`--method parallel`)
- Multiple replicas at different temperatures
- Periodic state swaps between adjacent temperatures
- Best for escaping deep local minima

---

## Installation

### Using Conda (Recommended)

```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate markov-queens

# Run solver
python main.py --size 3 --steps 10000
```

### Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run solver
python main.py --size 3 --steps 10000
```

### Dependencies
- Python 3.8+
- JAX (with optional GPU support)
- NumPy
- Matplotlib

---

## Usage

### Basic Usage

```bash
# Simple run (N=3, 10,000 steps)
python main.py --size 3 --steps 10000

# Improved method with geometric cooling
python main.py --size 4 --steps 50000 --method improved --cooling geometric

# Parallel tempering with 8 replicas
python main.py --size 4 --steps 20000 --method parallel --replicas 8

# Show interactive plots
python main.py --size 3 --steps 10000 --show
```

### Command Line Options

| Option | Description | Default | Choices |
|--------|-------------|---------|---------|
| `--size` | Board size N | 3 | Any positive integer |
| `--steps` | MCMC iterations | 10000 | Any positive integer |
| `--seed` | Random seed | 42 | Any integer |
| `--method` | MCMC method | basic | basic, improved, parallel |
| `--cooling` | Cooling schedule | geometric | linear, geometric, adaptive |
| `--beta-min` | Initial β | 0.1 | Any positive float |
| `--beta-max` | Final β | 50.0 | Any positive float |
| `--replicas` | Parallel tempering replicas | 8 | Any positive integer |
| `--show` | Show interactive plots | False | Flag |

### Example Runs

```bash
# Quick test (unsolvable N=3)
python main.py --size 3 --steps 10000 --method improved

# Try solvable N=11
python main.py --size 11 --steps 100000 --method improved --beta-max 100

# Parallel tempering for difficult cases
python main.py --size 4 --steps 50000 --method parallel --replicas 10
```

---

## Project Structure

```
Markov/
├── main.py                 # Main entry point
├── experiments.py          # Experiment scripts for project tasks
├── src/
│   ├── __init__.py
│   ├── board.py           # BoardState class, energy computation
│   ├── solver.py          # MCMCSolver with all methods
│   └── visualize.py       # 3D visualization functions
├── tests/
│   ├── __init__.py
│   └── test_implementation.py  # Unit tests
├── environment.yml         # Conda environment
├── requirements.txt        # pip requirements
└── README.md              # This file
```

### Key Components

| File | Description |
|------|-------------|
| `board.py` | `BoardState` class managing queen positions and energy |
| `solver.py` | `MCMCSolver` with basic, improved, and parallel tempering methods |
| `visualize.py` | 3D visualization with cube wireframe, attack lines |
| `experiments.py` | Scripts for running project experiments |

---

## Experiments

The `experiments.py` script runs experiments for the mini-project tasks:

```bash
# Run all experiments
python experiments.py

# Run specific experiment
python experiments.py 1  # Energy vs time
python experiments.py 2  # Simulated annealing comparison
python experiments.py 3  # Minimal energy vs N
python experiments.py 4  # Identify special N values
```

### Experiment 1: Energy vs Time
- Plots energy evolution averaged over multiple runs
- Shows convergence behavior

### Experiment 2: Simulated Annealing
- Compares linear, geometric, and adaptive cooling
- Determines best cooling schedule

### Experiment 3: Minimal Energy vs N
- Plots minimum energy as function of board size
- Shows scaling behavior

### Experiment 4: Special N Values
- Identifies N values with significantly lower energy
- Validates theoretical solvability conditions

---

## Results

### Output Files

| File | Description |
|------|-------------|
| `solution_{N}x{N}x{N}.png` | 3D visualization of final configuration |
| `energy_history.png` | Energy evolution during MCMC |
| `exp1_energy_vs_time_N{N}.png` | Experiment 1 results |
| `exp2_annealing_comparison_N{N}.png` | Experiment 2 results |
| `exp3_minimal_energy_vs_N.png` | Experiment 3 results |
| `exp4_special_N_values.png` | Experiment 4 results |

### Visualization Features

The 3D visualization includes:
- **Cube wireframe**: Shows board boundaries
- **Grid lines**: Cell boundaries (for N ≤ 10)
- **Green spheres**: Safe queens (no conflicts)
- **Red spheres**: Attacking queens
- **Dashed lines**: Attack connections
- **Statistics box**: Board info, density, conflicts
- **Solvability info**: gcd(N, 210) in title

### Performance

| N | Steps | Time | Rate | Best Energy |
|---|-------|------|------|-------------|
| 3 | 10,000 | ~3s | ~3,000/s | 13 |
| 4 | 50,000 | ~15s | ~3,500/s | 21 |
| 10 | 100,000 | ~25s | ~4,000/s | ~100 |
| 11 | 100,000 | ~25s | ~4,000/s | ~150 |

---

## Troubleshooting

### Conda Activation Error

If you see `CondaError: Run 'conda init' before 'conda activate'`:

```bash
# Initialize Conda
conda init zsh  # or bash

# Restart terminal, then activate
conda activate markov-queens
```

### JAX Installation Issues

```bash
# CPU-only installation
pip install jax jaxlib

# For GPU support, see: https://github.com/google/jax#installation
```

### Memory Issues for Large N

For large boards (N > 15), reduce memory usage:
```bash
# Use fewer steps
python main.py --size 15 --steps 50000

# Use parallel tempering with fewer replicas
python main.py --size 15 --steps 50000 --method parallel --replicas 4
```

### Slow Performance

1. Ensure JAX is using JIT compilation (first run is slow)
2. Use `--method improved` for better convergence
3. Increase `--beta-max` for faster cooling

---

## References

1. Klarner, D.A. (1967). "The Problem of Reflecting Queens"
2. Metropolis, N. et al. (1953). "Equation of State Calculations"
3. Kirkpatrick, S. et al. (1983). "Optimization by Simulated Annealing"
4. Swendsen, R.H. & Wang, J.S. (1986). "Replica Monte Carlo Simulation"

---

## License

This project is for educational purposes as part of the EPFL Markov Chains course.

---

## Author

EPFL - Markov Chains and Algorithmic Applications - Mini-Project
