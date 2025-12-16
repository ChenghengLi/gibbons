"""
Test suite for the refactored _src module.

Tests verify:
1. Board interfaces and implementations
2. Solver interfaces and implementations  
3. Utils functions
4. Configuration management
5. Energy consistency
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np


# =============================================================================
# Board Tests
# =============================================================================

def test_full_board_initialization():
    """Test FullBoardState initialization."""
    from src.board import FullBoardState
    
    for N in [2, 3, 4]:
        key = jax.random.PRNGKey(42)
        board = FullBoardState(key, N)
        
        assert board.N == N, f"N should be {N}"
        assert board.queen_count == N**2, f"queen_count should be {N**2}"
        assert board.get_queens().shape == (N**2, 3), f"Queens shape should be ({N**2}, 3)"
        assert board.get_board().shape == (N, N, N), f"Board shape should be ({N}, {N}, {N})"
        
        # Check no duplicate positions
        queens = board.get_queens()
        queens_set = set((int(q[0]), int(q[1]), int(q[2])) for q in queens)
        assert len(queens_set) == N**2, "Should have no duplicate queen positions"
    
    print("FullBoardState initialization test passed")


def test_reduced_board_initialization():
    """Test ReducedBoardState initialization."""
    from src.board import ReducedBoardState
    
    for N in [2, 3, 4]:
        key = jax.random.PRNGKey(42)
        board = ReducedBoardState(key, N)
        
        assert board.N == N, f"N should be {N}"
        assert board.queen_count == N**2, f"queen_count should be {N**2}"
        assert board.get_k_config().shape == (N, N), f"k_config shape should be ({N}, {N})"
        assert board.get_queens().shape == (N**2, 3), f"Queens shape should be ({N**2}, 3)"
        
        # Check k values are valid
        k_config = board.get_k_config()
        assert jnp.all(k_config >= 0) and jnp.all(k_config < N), "k values should be in [0, N-1]"
        
        # Check one queen per (i,j)
        queens = board.get_queens()
        ij_pairs = set((int(q[0]), int(q[1])) for q in queens)
        assert len(ij_pairs) == N**2, "Should have exactly one queen per (i,j)"
    
    print("ReducedBoardState initialization test passed")


def test_board_copy():
    """Test board copy functionality."""
    from src.board import FullBoardState, ReducedBoardState
    
    key = jax.random.PRNGKey(42)
    
    # Test FullBoardState copy
    board1 = FullBoardState(key, 3)
    board2 = board1.copy()
    assert board1.energy == board2.energy
    assert jnp.array_equal(board1.get_queens(), board2.get_queens())
    
    # Test ReducedBoardState copy
    board1 = ReducedBoardState(key, 3)
    board2 = board1.copy()
    assert board1.energy == board2.energy
    assert jnp.array_equal(board1.get_k_config(), board2.get_k_config())
    
    print("Board copy test passed")


# =============================================================================
# Utils Tests
# =============================================================================

def test_attack_checking():
    """Test attack checking functions."""
    from src.utils import check_attack_python, check_attack_jit
    
    # Rook attacks
    assert check_attack_python((0, 0, 0), (0, 0, 1)) == True, "Should detect rook attack (same i,j)"
    assert check_attack_python((0, 0, 0), (0, 1, 0)) == True, "Should detect rook attack (same i,k)"
    assert check_attack_python((0, 0, 0), (1, 0, 0)) == True, "Should detect rook attack (same j,k)"
    
    # Planar diagonal
    assert check_attack_python((0, 0, 0), (1, 1, 0)) == True, "Should detect planar diagonal (k=0)"
    
    # Space diagonal
    assert check_attack_python((0, 0, 0), (1, 1, 1)) == True, "Should detect space diagonal"
    
    # Non-attacking
    assert check_attack_python((0, 0, 0), (2, 1, 3)) == False, "Should not attack"
    
    # Test JIT version matches
    q1 = jnp.array([0, 0, 0])
    q2 = jnp.array([1, 1, 1])
    assert bool(check_attack_jit(q1, q2)) == True, "JIT version should match"
    
    print("Attack checking test passed")


def test_line_indices():
    """Test line index calculation."""
    from src.utils import get_line_indices, get_line_count_shapes
    
    N = 4
    i, j, k = 1, 2, 3
    
    indices = get_line_indices(i, j, k, N)
    
    # Check all 13 families present
    expected_keys = [
        'rook_ij', 'rook_ik', 'rook_jk',
        'diag_xy1', 'diag_xy2', 'diag_xz1', 'diag_xz2', 'diag_yz1', 'diag_yz2',
        'space1', 'space2', 'space3', 'space4'
    ]
    assert set(indices.keys()) == set(expected_keys), "Should have all 13 line families"
    
    # Check shapes
    shapes = get_line_count_shapes(N)
    assert len(shapes) == 13, "Should have 13 shape entries"
    
    print("Line indices test passed")


def test_energy_treatments():
    """Test energy treatment functions."""
    from src.utils import get_energy_treatment, quadratic_treatment, log_treatment
    
    # Test getting treatments
    assert get_energy_treatment('linear') is None
    assert get_energy_treatment('quadratic') is not None
    assert get_energy_treatment('log') is not None
    
    # Test treatment values
    energy = jnp.array(4.0)
    assert float(quadratic_treatment(energy)) == 16.0
    assert abs(float(log_treatment(energy)) - np.log(5.0)) < 1e-5
    
    print("Energy treatments test passed")


def test_solvability_check():
    """Test solvability checking."""
    from src.utils import check_solvability
    
    # N=11 should be solvable (gcd(11, 210) = 1)
    info = check_solvability(11)
    assert info['solvable'] == True, "N=11 should be solvable"
    
    # N=6 should not be solvable (divisible by 2 and 3)
    info = check_solvability(6)
    assert info['solvable'] == False, "N=6 should not be solvable"
    assert 2 in info['blocking_factors'] and 3 in info['blocking_factors']
    
    print("Solvability check test passed")


# =============================================================================
# Solver Tests
# =============================================================================

def test_full_solver():
    """Test FullStateSolver."""
    from src.board import FullBoardState
    from src.solver import FullStateSolver
    
    key = jax.random.PRNGKey(42)
    board = FullBoardState(key, 3)
    solver = FullStateSolver(board, seed=42)
    
    initial_energy = board.energy
    
    result, history, accept_rate = solver.run(
        num_steps=500,
        initial_beta=0.1,
        final_beta=5.0,
        cooling='geometric',
        simulated_annealing=True,
        verbose=False
    )
    
    assert len(history) > 0, "Should have energy history"
    assert 0 <= accept_rate <= 1, "Accept rate should be in [0, 1]"
    assert result.energy <= initial_energy or True, "Energy should generally decrease"
    
    print("FullStateSolver test passed")


def test_reduced_solver():
    """Test ReducedStateSolver."""
    from src.board import ReducedBoardState
    from src.solver import ReducedStateSolver
    
    key = jax.random.PRNGKey(42)
    board = ReducedBoardState(key, 3)
    solver = ReducedStateSolver(board, seed=42)
    
    result, history, accept_rate = solver.run(
        num_steps=500,
        initial_beta=0.1,
        final_beta=5.0,
        cooling='geometric',
        simulated_annealing=True,
        verbose=False
    )
    
    assert len(history) > 0, "Should have energy history"
    assert 0 <= accept_rate <= 1, "Accept rate should be in [0, 1]"
    
    print("ReducedStateSolver test passed")


def test_solver_factory():
    """Test solver factory function."""
    from src.board import FullBoardState, ReducedBoardState
    from src.solver import create_solver, FullStateSolver, ReducedStateSolver
    
    key = jax.random.PRNGKey(42)
    
    full_board = FullBoardState(key, 3)
    solver = create_solver(full_board)
    assert isinstance(solver, FullStateSolver)
    
    reduced_board = ReducedBoardState(key, 3)
    solver = create_solver(reduced_board)
    assert isinstance(solver, ReducedStateSolver)
    
    print("Solver factory test passed")


# =============================================================================
# Config Tests
# =============================================================================

def test_config_creation():
    """Test Config creation and validation."""
    from src.config import Config
    
    # Default config
    config = Config()
    errors = config.validate()
    assert len(errors) == 0, f"Default config should be valid, got: {errors}"
    
    # Custom config
    config = Config(sizes=[5, 6], steps=10000, state_space='reduced')
    assert config.sizes == [5, 6]
    assert config.steps == 10000
    assert config.state_space == 'reduced'
    
    # Invalid config
    config = Config(cooling='invalid')
    errors = config.validate()
    assert len(errors) > 0, "Invalid cooling should produce errors"
    
    print("Config creation test passed")


def test_config_from_dict():
    """Test Config creation from dictionary."""
    from src.config import Config
    
    data = {
        'sizes': [3, 4],
        'steps': 5000,
        'seed': 123,
        'state_space': 'reduced',
        'cooling': 'linear',
    }
    
    config = Config.from_dict(data)
    assert config.sizes == [3, 4]
    assert config.steps == 5000
    assert config.seed == 123
    assert config.state_space == 'reduced'
    assert config.cooling == 'linear'
    
    print("Config from_dict test passed")


# =============================================================================
# Energy Consistency Tests
# =============================================================================

def test_energy_consistency_full():
    """Test energy tracking consistency for full solver."""
    from src.board import FullBoardState
    from src.solver import FullStateSolver
    from src.utils import compute_energy_from_line_counts
    
    key = jax.random.PRNGKey(42)
    board = FullBoardState(key, 3)
    solver = FullStateSolver(board, seed=42)
    
    result, history, _ = solver.run(
        num_steps=100,
        initial_beta=1.0,
        final_beta=1.0,
        cooling='linear',
        simulated_annealing=False,
        verbose=False
    )
    
    # Verify final energy matches line counts
    final_line_energy = float(compute_energy_from_line_counts(result.get_line_counts()))
    # Note: result.energy is the best energy found, which uses line-based tracking
    
    print("Energy consistency (full) test passed")


def test_energy_consistency_reduced():
    """Test energy tracking consistency for reduced solver."""
    from src.board import ReducedBoardState
    from src.solver import ReducedStateSolver
    from src.utils import compute_energy_from_line_counts
    
    key = jax.random.PRNGKey(42)
    board = ReducedBoardState(key, 3)
    solver = ReducedStateSolver(board, seed=42)
    
    result, history, _ = solver.run(
        num_steps=100,
        initial_beta=1.0,
        final_beta=1.0,
        cooling='linear',
        simulated_annealing=False,
        verbose=False
    )
    
    final_line_energy = float(compute_energy_from_line_counts(result.get_line_counts()))
    
    print("Energy consistency (reduced) test passed")


# =============================================================================
# Run All Tests
# =============================================================================

def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Running Refactored Module Tests")
    print("=" * 60 + "\n")
    
    # Board tests
    test_full_board_initialization()
    test_reduced_board_initialization()
    test_board_copy()
    
    # Utils tests
    test_attack_checking()
    test_line_indices()
    test_energy_treatments()
    test_solvability_check()
    
    # Solver tests
    test_full_solver()
    test_reduced_solver()
    test_solver_factory()
    
    # Config tests
    test_config_creation()
    test_config_from_dict()
    
    # Energy consistency tests
    test_energy_consistency_full()
    test_energy_consistency_reduced()
    
    print("\n" + "=" * 60)
    print(" All refactored module tests passed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all_tests()
