import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np
import math


def check_attack(q1, q2):
    """
    Check if two queens attack each other.
    
    Based on Definition 2.2 (Attack Relation):
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


def draw_cube_wireframe(ax, N):
    """Draw the outer cube wireframe for the board."""
    # Define the vertices of the cube
    vertices = [
        [0, 0, 0], [N, 0, 0], [N, N, 0], [0, N, 0],  # Bottom face
        [0, 0, N], [N, 0, N], [N, N, N], [0, N, N]   # Top face
    ]
    
    # Define the edges
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ]
    
    # Draw edges
    for edge in edges:
        points = [vertices[edge[0]], vertices[edge[1]]]
        ax.plot3D(*zip(*points), color='black', linewidth=1.5, alpha=0.6)


def draw_grid_planes(ax, N, alpha=0.1):
    """Draw semi-transparent grid planes at each level."""
    # Only draw a few planes to avoid clutter
    for k in range(N + 1):
        # XY plane at z=k
        xx, yy = np.meshgrid([0, N], [0, N])
        zz = np.ones_like(xx) * k
        ax.plot_surface(xx, yy, zz, alpha=alpha, color='blue', edgecolor='none')


def draw_cell_grid(ax, N):
    """Draw grid lines to show individual cells."""
    # Draw grid lines on bottom face
    for i in range(N + 1):
        ax.plot([i, i], [0, N], [0, 0], color='gray', linewidth=0.5, alpha=0.3)
        ax.plot([0, N], [i, i], [0, 0], color='gray', linewidth=0.5, alpha=0.3)
    
    # Draw vertical lines at corners
    for i in range(N + 1):
        for j in range(N + 1):
            if (i == 0 or i == N) and (j == 0 or j == N):
                ax.plot([i, i], [j, j], [0, N], color='gray', linewidth=0.5, alpha=0.3)


def draw_attack_lines(ax, queens, attacking_pairs):
    """Draw lines between attacking queen pairs."""
    for i, j in attacking_pairs:
        q1, q2 = queens[i], queens[j]
        # Add 0.5 offset to center in cells
        ax.plot([q1[0]+0.5, q2[0]+0.5], [q1[1]+0.5, q2[1]+0.5], [q1[2]+0.5, q2[2]+0.5], 
                color='red', linewidth=0.8, alpha=0.3, linestyle='--')


def visualize_solution(state, endangered, filename=None):
    """
    Create enhanced 3D visualization of queen positions.
    
    Features:
    - 3D cube wireframe showing the board
    - Grid lines for cell boundaries
    - Green spheres: Safe queens (not attacking)
    - Red spheres: Attacking queens (in conflict)
    - Attack lines between conflicting queens
    - Solvability information in title
    """
    queens = np.array(state.queens)
    N = state.N
    energy = float(state.energy)
    
    # Check solvability
    gcd = math.gcd(N, 210)
    solvable = (gcd == 1)
    
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set background color
    ax.set_facecolor('white')
    
    # Draw the cube wireframe
    draw_cube_wireframe(ax, N)
    
    # Draw cell grid (only for small N)
    if N <= 10:
        draw_cell_grid(ax, N)
    
    # Identify attacking queens and pairs
    attacking_set = set()
    attacking_pairs = []
    
    for i in range(len(queens)):
        for j in range(i + 1, len(queens)):
            if check_attack(queens[i], queens[j]):
                attacking_set.add(i)
                attacking_set.add(j)
                attacking_pairs.append((i, j))
    
    # Separate safe and attacking queens
    safe_indices = [i for i in range(len(queens)) if i not in attacking_set]
    attacking_indices = list(attacking_set)
    
    safe = queens[safe_indices] if safe_indices else np.empty((0, 3))
    attacking = queens[attacking_indices] if attacking_indices else np.empty((0, 3))
    
    # Draw attack lines (only for small N or few attacks)
    if len(attacking_pairs) <= 50:
        draw_attack_lines(ax, queens, attacking_pairs)
    
    # Calculate marker size based on N
    marker_size = max(50, min(400, 2000 / N))
    
    # Plot safe queens in green with 3D effect
    if len(safe) > 0:
        ax.scatter(safe[:,0] + 0.5, safe[:,1] + 0.5, safe[:,2] + 0.5, 
                   s=marker_size, c='limegreen', marker='o', 
                   label=f'Safe Queens ({len(safe)})',
                   edgecolors='darkgreen', linewidths=1.5, alpha=0.9,
                   depthshade=True)
    
    # Plot attacking queens in red with 3D effect
    if len(attacking) > 0:
        ax.scatter(attacking[:,0] + 0.5, attacking[:,1] + 0.5, attacking[:,2] + 0.5, 
                   s=marker_size, c='tomato', marker='o', 
                   label=f'Attacking Queens ({len(attacking)})',
                   edgecolors='darkred', linewidths=1.5, alpha=0.9,
                   depthshade=True)
    
    # Set labels and title
    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z', fontsize=12, fontweight='bold')
    
    # Create informative title
    status = "SOLVED!" if energy == 0 else f"Energy: {energy:.0f} conflicts"
    solvable_str = "Solvable" if solvable else "Unsolvable"
    title = f'3D N²-Queens Problem: {N}×{N}×{N} Board\n'
    title += f'{N**2} Queens | {status} | gcd({N},210)={gcd} ({solvable_str})'
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    # Set axis limits with padding
    ax.set_xlim(-0.2, N + 0.2)
    ax.set_ylim(-0.2, N + 0.2)
    ax.set_zlim(-0.2, N + 0.2)
    
    # Set integer ticks
    ax.set_xticks(range(N + 1))
    ax.set_yticks(range(N + 1))
    ax.set_zticks(range(N + 1))
    
    # Add legend with statistics
    legend = ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # Add text box with statistics
    stats_text = f'Board: {N}³ = {N**3} cells\n'
    stats_text += f'Queens: {N**2}\n'
    stats_text += f'Density: {N**2/N**3:.1%}\n'
    stats_text += f'Conflicts: {int(energy)}'
    stats_text += f'Endangered Queens: {endangered}'
    
    ax.text2D(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
              verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return filename
    return fig


def plot_energy_history(energy_history, filename=None):
    """Plot energy evolution during MCMC"""
    plt.figure(figsize=(10, 6))
    plt.plot(energy_history)
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.title('Energy Evolution During MCMC')
    plt.grid(True)
    
    if filename:
        plt.savefig(filename)
        plt.close()
        return filename
    return plt.gcf()


def visualize_latin_square(state, endangered, filename=None):
    """
    Visualize the 3D Queens solution as a Latin square projection.
    
    Shows an N×N grid where each cell displays the k-coordinate (layer)
    of the queen at that (i,j) position. Empty cells and conflicts are marked.
    """
    queens = np.array(state.queens)
    N = state.N
    energy = float(state.energy)
    
    # Create N×N grid to store k-coordinates
    grid = np.full((N, N), -1, dtype=int)  # -1 means no queen
    conflicts = np.zeros((N, N), dtype=bool)
    
    # Populate grid
    for q in queens:
        i, j, k = int(q[0]), int(q[1]), int(q[2])
        if grid[i, j] != -1:
            # Multiple queens at same (i,j) - mark as conflict
            conflicts[i, j] = True
        else:
            grid[i, j] = k
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create colormap
    cmap = plt.cm.viridis
    
    # Display grid
    for i in range(N):
        for j in range(N):
            if conflicts[i, j]:
                # Conflict cell - red
                ax.add_patch(plt.Rectangle((j, N-1-i), 1, 1, facecolor='red', edgecolor='black', linewidth=1))
                ax.text(j+0.5, N-1-i+0.5, 'X', ha='center', va='center', fontsize=14, color='white', fontweight='bold')
            elif grid[i, j] == -1:
                # Empty cell - light gray
                ax.add_patch(plt.Rectangle((j, N-1-i), 1, 1, facecolor='lightgray', edgecolor='black', linewidth=1))
            else:
                # Queen present - color by k-coordinate
                k_val = grid[i, j]
                color = cmap(k_val / (N - 1)) if N > 1 else cmap(0.5)
                ax.add_patch(plt.Rectangle((j, N-1-i), 1, 1, facecolor=color, edgecolor='black', linewidth=1))
                ax.text(j+0.5, N-1-i+0.5, str(k_val), ha='center', va='center', fontsize=12, color='white', fontweight='bold')
    
    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels(np.arange(N))
    ax.set_yticklabels(np.arange(N-1, -1, -1))
    ax.set_xlabel('j (column)', fontsize=12, fontweight='bold')
    ax.set_ylabel('i (row)', fontsize=12, fontweight='bold')
    ax.grid(False)
    
    # Title
    status = "SOLVED!" if energy == 0 else f"Energy: {energy:.0f} conflicts"
    title = f'Latin Square Projection: {N}×{N}×{N} Board\n'
    title += f'Cell (i,j) shows k-coordinate of queen | {status}'
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=N-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, fraction=0.046)
    cbar.set_label('k (layer)', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return filename
    return fig


def plot_averaged_energy_history(histories, filename=None, metadata=None):
    """
    Plot averaged energy evolution across multiple runs with metadata.
    
    metadata: dict containing:
    - beta_range: (start, end) or 'Constant'
    - steps: int
    - cooling_method: str
    - board_size: int
    - final_energy: float (average)
    """
    # Convert to numpy array if not already
    histories_arr = np.array(histories)
    
    mean_energy = np.mean(histories_arr, axis=0)
    std_energy = np.std(histories_arr, axis=0)
    
    plt.figure(figsize=(10, 7))  # Slightly taller for subtitle
    x = np.arange(len(mean_energy))
    
    # Plot individual runs first (so they are in background)
    # Generate distinct colors for each run
    num_runs = len(histories)
    colors = plt.cm.jet(np.linspace(0, 1, num_runs))
    
    for i, h in enumerate(histories):
        plt.plot(h, color=colors[i], alpha=0.3, linewidth=0.8, label=f'_Run {i+1}')
    
    # Add a dummy artist for the legend to represent individual runs
    plt.plot([], [], color='gray', alpha=0.5, linewidth=0.8, label=f'Individual Runs ({num_runs})')
    
    # Plot average on top
    plt.plot(x, mean_energy, label='Average Energy', color='black', linewidth=1.6, zorder=10)
    plt.fill_between(x, mean_energy - std_energy, mean_energy + std_energy, color='black', alpha=0.1, label='±1 Std Dev', zorder=5)
        
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    
    title = f'Averaged Energy Evolution ({len(histories)} runs)'
    
    if metadata:
        beta_info = f"β: {metadata['beta_range']}"
        config_info = f"N={metadata['board_size']}, Steps={metadata['steps']}, Cooling={metadata['cooling_method']}"
        result_info = f"Avg Final Energy: {metadata['final_energy']:.2f}"
        
        subtitle = f"{config_info}\n{beta_info} | {result_info}"
        plt.title(title + '\n' + subtitle, fontsize=11)
        
        # Add a text box with details
        textstr = '\n'.join((
            f"Board Size: {metadata['board_size']}x{metadata['board_size']}x{metadata['board_size']}",
            f"Steps: {metadata['steps']}",
            f"Cooling: {metadata['cooling_method']}",
            f"Beta: {metadata['beta_range']}",
            f"Avg Final Energy: {metadata['final_energy']:.2f}"
        ))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.gca().text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=9,
                       verticalalignment='top', bbox=props)
    else:
        plt.title(title)
        
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename)
        plt.close()
        return filename
    return plt.gcf()
