# 3D N² Queens MCMC Solver

This project solves the 3D N²-Queens problem using Markov Chain Monte Carlo (MCMC) with JAX. The goal is to place N² queens on an N×N×N cubic board such that no two queens attack each other.

## Problem Description

Based on the theoretical formulation:

- **Board**: B_N = {0, 1, ..., N-1}³ (a discrete cube with N³ cells)
- **Configuration**: A set of N² queen positions
- **State Space**: |S| = C(N³, N²) possible configurations

### Attack Relations (Definition 2.2)

Two queens at positions (i,j,k) and (i',j',k') attack each other if:

1. **Rook-type**: Share at least two coordinates
   - Same (i,j), same (i,k), or same (j,k)

2. **Planar diagonal**: On a diagonal within a coordinate plane
   - xy-plane: k=k' and |i-i'|=|j-j'|≠0
   - xz-plane: j=j' and |i-i'|=|k-k'|≠0
   - yz-plane: i=i' and |j-j'|=|k-k'|≠0

3. **Space diagonal**: All coordinate differences equal
   - |i-i'|=|j-j'|=|k-k'|≠0

### Energy Function (Definition 2.5)

J(s) = number of attacking pairs in configuration s

A valid solution has J(s) = 0.

## Features
- **Metropolis-Hastings Algorithm**: Efficient solution space exploration
- **Attack Visualization**: 
  - Red queens: Attacking other queens
  - Green queens: Safe/non-attacking
- **Energy Tracking**: Plots energy evolution during MCMC
- **Simulated Annealing**: Gradually increases search focus
- **3D Visualization**: Interactive plot of queen positions

## Requirements
- Python 3.7+
- `pip install -r requirements.txt`

## Usage
```bash
# Basic usage (N=3, 10,000 steps)
python main.py --size 3 --steps 10000

# Larger board (N=4, 50,000 steps)
python main.py --size 4 --steps 50000

# Custom random seed
python main.py --size 3 --steps 10000 --seed 123
```

### Command Line Options:
| Option    | Description                  | Default |
|-----------|------------------------------|---------|
| `--size`  | Board size (N)               | 3       |
| `--steps` | MCMC iterations              | 10000   |
| `--seed`  | Random seed                  | 42      |

## Outputs
1. `solution_{N}x{N}x{N}.png` - 3D visualization:
   - Attacking queens shown in red
   - Safe queens shown in green
   - Energy value in title
2. `energy_history.png` - Energy evolution plot

## Examples
```bash
# Solve 3x3x3 board
python main.py --size 3 --steps 10000

# Solve 4x4x4 board with more iterations
python main.py --size 4 --steps 50000
```

## Visualization Example
![3x3x3 Solution](solution_3x3x3.png)

## Performance Tips
1. Start with small boards (N=3)
2. Increase steps for larger boards:
   - N=3: 10,000 steps
   - N=4: 50,000-100,000 steps
   - N=5: 100,000+ steps
3. Use different seeds if solution isn't found

## Troubleshooting
- Install requirements: `pip install -r requirements.txt`
- For JAX installation issues, see [JAX documentation](https://github.com/google/jax#installation)
### Conda Activation Error
If you see `CondaError: Run 'conda init' before 'conda activate'`:

1. Initialize Conda for your shell (replace `zsh` with your shell if different):
   ```bash
   conda init zsh
   ```
2. Restart your terminal or source the configuration file:
   ```bash
   source ~/.zshrc
   ```
3. Activate the environment:
   ```bash
   conda activate markov-queens
   ```
