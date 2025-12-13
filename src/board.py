import jax
import jax.numpy as jnp
import numpy as np

class BoardState:
    """
    Represents the 3D queens board state.
    
    Based on the theoretical formulation:
    - Board: B_N = {0, 1, ..., N-1}^3
    - Configuration: s = {(i_1,j_1,k_1), ..., (i_{N^2},j_{N^2},k_{N^2})} with |s| = N^2
    - State space: S = {s ⊂ B_N : |s| = N^2}, |S| = C(N^3, N^2)

    self.N
    self.queen_count
    self.total_cells
    self.queens
    self.board
    self.line_counts
    self.energy
    """
    def __init__(self, key, N):
        self.N = N
        self.queen_count = N**2
        self.total_cells = N**3
        self.initialize_state(key)
        
    def initialize_state(self, key):
        """Initialize random state with N^2 queens"""
        key, subkey = jax.random.split(key)
        flat_indices = jax.random.choice(
            subkey, self.total_cells, shape=(self.queen_count,), replace=False
        )
        self.queens = jnp.array(np.unravel_index(flat_indices, (self.N, self.N, self.N))).T
        
        # Create board (occupancy grid)
        self.board = jnp.zeros((self.N, self.N, self.N), dtype=bool)
        self.board = self.board.at[self.queens[:,0], self.queens[:,1], self.queens[:,2]].set(True)

        # Initialize and populate line counts
        self.line_counts = self._create_line_counts()
        self._populate_line_counts()
        self.energy = self._compute_energy()
    
    def _populate_line_counts(self):
        """Populate line counts from current queen positions"""
        for q in self.queens:
            i, j, k = int(q[0]), int(q[1]), int(q[2])
            sigs = self._get_line_signatures_indexed(i, j, k)
            for line_type, sig in sigs:
                self.line_counts[line_type] = self.line_counts[line_type].at[sig].add(1)
    
    def _get_line_signatures_indexed(self, i, j, k):
        """
        Compute all 13 line signatures with pre-shifted indices.
        
        Based on Definition 2.6 (Attack Lines):
        - Rook lines: 3 families indexed by (i,j), (i,k), (j,k)
        - Planar diagonals: 6 families, two per plane
        - Space diagonals: 4 families
        """
        N = self.N
        return [
            ('rook_ij', (i, j)),
            ('rook_ik', (i, k)),
            ('rook_jk', (j, k)),
            ('diag_xy1', (k, i - j + N - 1)),  # (k, i-j) shifted
            ('diag_xy2', (k, i + j)),           # (k, i+j)
            ('diag_xz1', (j, i - k + N - 1)),  # (j, i-k) shifted
            ('diag_xz2', (j, i + k)),           # (j, i+k)
            ('diag_yz1', (i, j - k + N - 1)),  # (i, j-k) shifted
            ('diag_yz2', (i, j + k)),           # (i, j+k)
            ('space1', i - j + k + N - 1),      # i-j+k shifted
            ('space2', i - j - k + 2 * (N - 1)), # i-j-k shifted
            ('space3', i + j - k + N - 1),      # i+j-k shifted
            ('space4', i + j + k),              # i+j+k
        ]
        
    def _create_line_counts(self):
        """Create empty line count structures"""
        return {
            'rook_ij': jnp.zeros((self.N, self.N)),
            'rook_ik': jnp.zeros((self.N, self.N)),
            'rook_jk': jnp.zeros((self.N, self.N)),
            'diag_xy1': jnp.zeros((self.N, 2*self.N-1)),
            'diag_xy2': jnp.zeros((self.N, 2*self.N-1)),
            'diag_xz1': jnp.zeros((self.N, 2*self.N-1)),
            'diag_xz2': jnp.zeros((self.N, 2*self.N-1)),
            'diag_yz1': jnp.zeros((self.N, 2*self.N-1)),
            'diag_yz2': jnp.zeros((self.N, 2*self.N-1)),
            'space1': jnp.zeros(3*self.N-2),
            'space2': jnp.zeros(3*self.N-2),
            'space3': jnp.zeros(3*self.N-2),
            'space4': jnp.zeros(3*self.N-2)
        }
        
    def _get_line_signatures(self, i, j, k):
        """Compute all 13 line signatures for a position"""
        return [
            ('rook_ij', (i, j)),
            ('rook_ik', (i, k)),
            ('rook_jk', (j, k)),
            ('diag_xy1', (k, i - j)), 
            ('diag_xy2', (k, i + j)),
            ('diag_xz1', (j, i - k)),
            ('diag_xz2', (j, i + k)),
            ('diag_yz1', (i, j - k)),
            ('diag_yz2', (i, j + k)),
            ('space1', i - j + k),
            ('space2', i - j - k),
            ('space3', i + j - k),
            ('space4', i + j + k)
        ]
        
    def _compute_energy(self):
        """
        Compute current energy by counting attacking pairs directly.
        
        Note: We cannot simply sum over all line families because
        a pair of queens can attack via multiple types simultaneously
        (e.g., rook-type attack counts on multiple rook lines).

        Instead, we count each attacking pair exactly once.
        """
        count = 0
        n = len(self.queens)
        for i in range(n):
            for j in range(i + 1, n):
                if self._check_attack(self.queens[i], self.queens[j]):
                    count += 1
        return jnp.array(float(count))
    
    def _check_attack(self, q1, q2):
        """
        Check if two queens q1, q2, attack each other.
        
        Attack relation:
        - Rook-type: share at least two coordinates
        - Planar diagonal: |i-i'| = |j-j'| != 0 when k=k' (and analogous)
        - Space diagonal: |i-i'| = |j-j'| = |k-k'| != 0
        """
        i1, j1, k1 = int(q1[0]), int(q1[1]), int(q1[2])
        i2, j2, k2 = int(q2[0]), int(q2[1]), int(q2[2])
        
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