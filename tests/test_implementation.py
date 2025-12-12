"""
Test suite to verify implementation matches the theoretical specification.

Based on the LaTeX document definitions:
- Definition 2.1: Board B_N = {0,1,...,N-1}^3
- Definition 2.2: Attack relations (rook, planar diagonal, space diagonal)
- Definition 2.3: Configuration s with |s| = N^2
- Definition 2.4: State space S with |S| = C(N^3, N^2)
- Definition 2.5: Energy J(s) = number of attacking pairs
- Algorithm 1: Efficient energy computation using 13 line families
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
from math import comb


def check_attack(q1, q2):
    """
    Check if two queens attack each other.
    
    Based on Definition 2.2 (Attack Relation):
    - Rook-type: share at least two coordinates
    - Planar diagonal: |i-i'| = |j-j'| != 0 when k=k' (and analogous)
    - Space diagonal: |i-i'| = |j-j'| = |k-k'| != 0
    """
    i1, j1, k1 = q1
    i2, j2, k2 = q2
    
    di = abs(i1 - i2)
    dj = abs(j1 - j2)
    dk = abs(k1 - k2)
    
    # Rook-type: share at least two coordinates
    if (i1 == i2 and j1 == j2) or (i1 == i2 and k1 == k2) or (j1 == j2 and k1 == k2):
        return True
    
    # Planar diagonal in xy-plane (k = k')
    if k1 == k2 and di == dj and di != 0:
        return True
    
    # Planar diagonal in xz-plane (j = j')
    if j1 == j2 and di == dk and di != 0:
        return True
    
    # Planar diagonal in yz-plane (i = i')
    if i1 == i2 and dj == dk and dj != 0:
        return True
    
    # Space diagonal: all three differences equal and non-zero
    if di == dj == dk and di != 0:
        return True
    
    return False


def count_attacking_pairs_naive(queens):
    """
    Naive O(N^4) algorithm to count attacking pairs.
    Used as ground truth for testing.
    
    Based on Definition 2.5:
    end(s) = sum over {q,q'} in s of 1[q attacks q']
    """
    count = 0
    n = len(queens)
    for i in range(n):
        for j in range(i + 1, n):
            if check_attack(queens[i], queens[j]):
                count += 1
    return count


def test_example_2_1():
    """
    Test Example 2.1 (Central Queen):
    On a 3x3x3 board, queen at (1,1,1) attacks:
    - (0,0,0) and (2,2,2) via space diagonals
    - (0,1,1) and (2,1,1) via rook-type (shared j,k)
    - (0,0,1) and (2,2,1) via planar diagonals in k=1 plane
    """
    center = (1, 1, 1)
    
    # Space diagonal attacks
    assert check_attack(center, (0, 0, 0)), "Should attack (0,0,0) via space diagonal"
    assert check_attack(center, (2, 2, 2)), "Should attack (2,2,2) via space diagonal"
    
    # Rook-type attacks (shared j,k)
    assert check_attack(center, (0, 1, 1)), "Should attack (0,1,1) via rook-type"
    assert check_attack(center, (2, 1, 1)), "Should attack (2,1,1) via rook-type"
    
    # Planar diagonal in k=1 plane
    assert check_attack(center, (0, 0, 1)), "Should attack (0,0,1) via planar diagonal"
    assert check_attack(center, (2, 2, 1)), "Should attack (2,2,1) via planar diagonal"
    
    print("✅ Example 2.1 (Central Queen) passed")


def test_example_2_2():
    """
    Test Example 2.2 (Non-Attacking Pair):
    Queens at (0,0,0) and (2,1,3) do not attack:
    - No two coordinates match (not rook-type)
    - Coordinate differences 2,1,3 are pairwise distinct (not planar)
    - Not all equal (not space diagonal)
    """
    q1 = (0, 0, 0)
    q2 = (2, 1, 3)
    
    assert not check_attack(q1, q2), "Queens at (0,0,0) and (2,1,3) should NOT attack"
    print("✅ Example 2.2 (Non-Attacking Pair) passed")


def test_example_2_3():
    """
    Test Example 2.3 (State Space Size):
    For N=3: |S| = C(27, 9) = 4,686,825
    For N=4: |S| = C(64, 16) ≈ 4.9 × 10^13
    """
    assert comb(27, 9) == 4686825, "State space size for N=3 should be 4,686,825"
    assert comb(64, 16) == 488526937079580, "State space size for N=4"
    print("✅ Example 2.3 (State Space Size) passed")


def test_example_2_4():
    """
    Test Example 2.4 (Energy Calculation):
    For N=2, queens at (0,0,0), (0,0,1), (1,1,0), (1,1,1):
    
    All 6 pairs attack each other:
    - (0,0,0)-(0,0,1): rook-type (shared i,j)
    - (0,0,0)-(1,1,0): planar diagonal in k=0 plane
    - (0,0,0)-(1,1,1): space diagonal
    - (0,0,1)-(1,1,0): space diagonal
    - (0,0,1)-(1,1,1): planar diagonal in k=1 plane
    - (1,1,0)-(1,1,1): rook-type (shared i,j)
    
    Note: The LaTeX document only counted rook-type attacks (2),
    but the correct energy is 6 (all pairs attack).
    """
    queens = [(0, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1)]
    energy = count_attacking_pairs_naive(queens)
    # All 6 pairs attack each other
    assert energy == 6, f"Energy should be 6, got {energy}"
    print("✅ Example 2.4 (Energy Calculation) passed - corrected to 6 pairs")


def test_line_signatures():
    """
    Test Definition 2.6 (Attack Lines):
    Verify the 13 line families are correctly computed.
    
    For position (i, j, k):
    - Rook lines: (i,j), (i,k), (j,k)
    - Planar diagonals: (k, i-j), (k, i+j), (j, i-k), (j, i+k), (i, j-k), (i, j+k)
    - Space diagonals: i-j+k, i-j-k, i+j-k, i+j+k
    """
    from src.board import BoardState
    
    # Create a dummy state to access the method
    key = jax.random.PRNGKey(0)
    state = BoardState(key, 3)
    N = 3
    
    # Test for position (1, 2, 0)
    i, j, k = 1, 2, 0
    sigs = state._get_line_signatures_indexed(i, j, k)
    
    # Expected values with pre-shifted indices for N=3
    expected = [
        ('rook_ij', (1, 2)),
        ('rook_ik', (1, 0)),
        ('rook_jk', (2, 0)),
        ('diag_xy1', (0, 1 - 2 + 2)),  # (k, i-j+N-1) = (0, 1)
        ('diag_xy2', (0, 1 + 2)),       # (k, i+j) = (0, 3)
        ('diag_xz1', (2, 1 - 0 + 2)),  # (j, i-k+N-1) = (2, 3)
        ('diag_xz2', (2, 1 + 0)),       # (j, i+k) = (2, 1)
        ('diag_yz1', (1, 2 - 0 + 2)),  # (i, j-k+N-1) = (1, 4)
        ('diag_yz2', (1, 2 + 0)),       # (i, j+k) = (1, 2)
        ('space1', 1 - 2 + 0 + 2),      # i-j+k+N-1 = 1
        ('space2', 1 - 2 - 0 + 4),      # i-j-k+2*(N-1) = 3
        ('space3', 1 + 2 - 0 + 2),      # i+j-k+N-1 = 5
        ('space4', 1 + 2 + 0),          # i+j+k = 3
    ]
    
    for (exp_type, exp_sig), (got_type, got_sig) in zip(expected, sigs):
        assert exp_type == got_type, f"Line type mismatch: {exp_type} vs {got_type}"
        assert exp_sig == got_sig, f"Signature mismatch for {exp_type}: {exp_sig} vs {got_sig}"
    
    print("✅ Line signatures test passed")


def test_energy_computation():
    """
    Test that the efficient O(N^2) energy computation matches the naive O(N^4) method.
    Based on Algorithm 1 and Proposition 2.1.
    """
    from src.board import BoardState
    
    for seed in range(5):
        key = jax.random.PRNGKey(seed)
        state = BoardState(key, 3)
        
        # Convert queens to list of tuples
        queens_list = [(int(q[0]), int(q[1]), int(q[2])) for q in state.queens]
        
        # Compute energy using naive method
        naive_energy = count_attacking_pairs_naive(queens_list)
        
        # Get energy from state (efficient method)
        efficient_energy = float(state.energy)
        
        assert naive_energy == efficient_energy, \
            f"Energy mismatch: naive={naive_energy}, efficient={efficient_energy}"
    
    print("✅ Energy computation test passed (naive vs efficient)")


def test_proposal_symmetry():
    """
    Test Proposition 3.1 (Symmetry of the Proposal):
    The proposal distribution should be symmetric.
    
    For configurations s and s' differing by a single queen:
    ψ(s→s') = ψ(s'→s) = 1/N^2 × 1/(N^3 - N^2)
    """
    N = 3
    # Probability of selecting one queen and one empty cell
    prob = (1 / N**2) * (1 / (N**3 - N**2))
    
    # This should be the same in both directions
    expected_prob = 1 / (9 * 18)  # For N=3: 1/162
    
    assert abs(prob - expected_prob) < 1e-10, \
        f"Proposal probability mismatch: {prob} vs {expected_prob}"
    
    print("✅ Proposal symmetry test passed")


def test_acceptance_probability():
    """
    Test Definition 3.2 (Acceptance Probability):
    α(s→s') = min(1, exp(-β × ΔJ))
    
    Example 3.1: β=2, J(s)=5, J(s')=7, ΔJ=2
    α = exp(-2 × 2) = e^(-4) ≈ 0.018
    """
    import math
    
    beta = 2
    delta_J = 2
    
    alpha = min(1, math.exp(-beta * delta_J))
    expected = math.exp(-4)
    
    assert abs(alpha - expected) < 1e-10, \
        f"Acceptance probability mismatch: {alpha} vs {expected}"
    assert abs(alpha - 0.018315638888734) < 1e-6, \
        f"Acceptance probability should be ≈ 0.018"
    
    print("✅ Acceptance probability test passed")


def test_board_initialization():
    """
    Test that board initialization creates:
    - N^2 queens
    - Valid positions in {0,...,N-1}^3
    - No duplicate positions
    """
    from src.board import BoardState
    
    for N in [2, 3, 4]:
        key = jax.random.PRNGKey(42)
        state = BoardState(key, N)
        
        # Check queen count
        assert len(state.queens) == N**2, \
            f"Should have {N**2} queens, got {len(state.queens)}"
        
        # Check positions are valid
        for q in state.queens:
            assert 0 <= q[0] < N and 0 <= q[1] < N and 0 <= q[2] < N, \
                f"Invalid position: {q}"
        
        # Check no duplicates
        queens_set = set((int(q[0]), int(q[1]), int(q[2])) for q in state.queens)
        assert len(queens_set) == N**2, "Duplicate queen positions found"
    
    print("✅ Board initialization test passed")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("Running Implementation Tests")
    print("Based on LaTeX Theoretical Specification")
    print("="*60 + "\n")
    
    test_example_2_1()
    test_example_2_2()
    test_example_2_3()
    test_example_2_4()
    test_line_signatures()
    test_energy_computation()
    test_proposal_symmetry()
    test_acceptance_probability()
    test_board_initialization()
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()
