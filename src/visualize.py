"""
Visualization functions for 3D N² Queens MCMC Solver.

This module provides:
- 3D solution visualization with attack lines
- Latin square projection visualization
- Energy history plots (single and averaged)
- Save functionality with metadata
- Competition format output (N² rows of x,y,z coordinates)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import json
from datetime import datetime


def check_attack(q1: Tuple, q2: Tuple) -> bool:
    """
    Check if two queens attack each other.
    
    Attack types:
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
    
    # Space diagonal
    if di == dj == dk and di != 0:
        return True
    
    return False


def draw_cube_wireframe(ax, N: int) -> None:
    """Draw the outer cube wireframe for the board."""
    vertices = [
        [0, 0, 0], [N, 0, 0], [N, N, 0], [0, N, 0],  # Bottom face
        [0, 0, N], [N, 0, N], [N, N, N], [0, N, N]   # Top face
    ]
    
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ]
    
    for edge in edges:
        points = [vertices[edge[0]], vertices[edge[1]]]
        ax.plot3D(*zip(*points), color='black', linewidth=1.5, alpha=0.6)


def draw_cell_grid(ax, N: int) -> None:
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


def draw_attack_lines(ax, queens: np.ndarray, attacking_pairs: List[Tuple[int, int]]) -> None:
    """Draw lines between attacking queen pairs."""
    for i, j in attacking_pairs:
        q1, q2 = queens[i], queens[j]
        ax.plot(
            [q1[0]+0.5, q2[0]+0.5], 
            [q1[1]+0.5, q2[1]+0.5], 
            [q1[2]+0.5, q2[2]+0.5],
            color='red', linewidth=0.8, alpha=0.3, linestyle='--'
        )


def count_endangered_queens(board) -> int:
    """Count queens that are in attacking positions."""
    queens = np.array(board.get_queens())
    attacking_set = set()
    
    for i in range(len(queens)):
        for j in range(i + 1, len(queens)):
            if check_attack(queens[i], queens[j]):
                attacking_set.add(i)
                attacking_set.add(j)
    
    return len(attacking_set)


def visualize_solution(
    board,
    filename: Optional[str] = None,
    show: bool = False,
    metadata: Optional[Dict] = None
) -> Optional[str]:
    """
    Create enhanced 3D visualization of queen positions.
    
    Features:
    - 3D cube wireframe showing the board
    - Grid lines for cell boundaries
    - Green spheres: Safe queens (not attacking)
    - Red spheres: Attacking queens (in conflict)
    - Attack lines between conflicting queens
    - Solvability information in title
    
    Args:
        board: Board state with get_queens(), N, energy attributes
        filename: Optional path to save the figure
        show: Whether to display the plot
        metadata: Optional dict with run parameters
        
    Returns:
        Filename if saved, None otherwise
    """
    queens = np.array(board.get_queens())
    N = board.N
    energy = float(board.energy)
    
    # Check solvability
    gcd = math.gcd(N, 210)
    solvable = (gcd == 1)
    
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection='3d')
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
    
    # Plot safe queens in green
    if len(safe) > 0:
        ax.scatter(
            safe[:,0] + 0.5, safe[:,1] + 0.5, safe[:,2] + 0.5,
            s=marker_size, c='limegreen', marker='o',
            label=f'Safe Queens ({len(safe)})',
            edgecolors='darkgreen', linewidths=1.5, alpha=0.9,
            depthshade=True
        )
    
    # Plot attacking queens in red
    if len(attacking) > 0:
        ax.scatter(
            attacking[:,0] + 0.5, attacking[:,1] + 0.5, attacking[:,2] + 0.5,
            s=marker_size, c='tomato', marker='o',
            label=f'Attacking Queens ({len(attacking)})',
            edgecolors='darkred', linewidths=1.5, alpha=0.9,
            depthshade=True
        )
    
    # Set labels
    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z', fontsize=12, fontweight='bold')
    
    # Create informative title
    status = "SOLVED!" if energy == 0 else f"Energy: {energy:.0f} conflicts"
    solvable_str = "Solvable" if solvable else "Unsolvable"
    title = f'3D N²-Queens Problem: {N}×{N}×{N} Board\n'
    title += f'{N**2} Queens | {status} | gcd({N},210)={gcd} ({solvable_str})'
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    # Set axis limits
    ax.set_xlim(-0.2, N + 0.2)
    ax.set_ylim(-0.2, N + 0.2)
    ax.set_zlim(-0.2, N + 0.2)
    
    # Set integer ticks
    ax.set_xticks(range(N + 1))
    ax.set_yticks(range(N + 1))
    ax.set_zticks(range(N + 1))
    
    # Add legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # Add text box with statistics
    stats_text = f'Board: {N}×{N}×{N}\n'
    stats_text += f'Queens: {N**2}\n'
    stats_text += f'Density: {N**2/N**3:.1%}\n'
    stats_text += f'Conflicts: {int(energy)}\n'
    stats_text += f'Safe: {len(safe)}\n'
    stats_text += f'Attacking: {len(attacking)}'
    
    if metadata:
        stats_text += f'\n\nState Space: {metadata.get("state_space", "N/A")}'
        stats_text += f'\nSteps: {metadata.get("steps", "N/A"):,}'
        stats_text += f'\nCooling: {metadata.get("cooling", "N/A")}'
    
    ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
        if not show:
            plt.close()
        return filename
    
    if show:
        plt.show()
    
    return None


def visualize_latin_square(
    board,
    filename: Optional[str] = None,
    show: bool = False,
    metadata: Optional[Dict] = None
) -> Optional[str]:
    """
    Visualize the 3D Queens solution as a Latin square projection.
    
    Shows an N×N grid where each cell displays the k-coordinate (layer)
    of the queen at that (i,j) position.
    
    Args:
        board: Board state
        filename: Optional path to save the figure
        show: Whether to display the plot
        metadata: Optional dict with run parameters
        
    Returns:
        Filename if saved, None otherwise
    """
    queens = np.array(board.get_queens())
    N = board.N
    energy = float(board.energy)
    
    # Create N×N grid to store k-coordinates
    grid = np.full((N, N), -1, dtype=int)
    conflicts = np.zeros((N, N), dtype=bool)
    
    # Populate grid
    for q in queens:
        i, j, k = int(q[0]), int(q[1]), int(q[2])
        if grid[i, j] != -1:
            conflicts[i, j] = True
        else:
            grid[i, j] = k
    
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.cm.viridis
    
    # Display grid
    for i in range(N):
        for j in range(N):
            if conflicts[i, j]:
                ax.add_patch(plt.Rectangle(
                    (j, N-1-i), 1, 1, 
                    facecolor='red', edgecolor='black', linewidth=1
                ))
                ax.text(j+0.5, N-1-i+0.5, 'X', ha='center', va='center',
                       fontsize=14, color='white', fontweight='bold')
            elif grid[i, j] == -1:
                ax.add_patch(plt.Rectangle(
                    (j, N-1-i), 1, 1,
                    facecolor='lightgray', edgecolor='black', linewidth=1
                ))
            else:
                k_val = grid[i, j]
                color = cmap(k_val / (N - 1)) if N > 1 else cmap(0.5)
                ax.add_patch(plt.Rectangle(
                    (j, N-1-i), 1, 1,
                    facecolor=color, edgecolor='black', linewidth=1
                ))
                ax.text(j+0.5, N-1-i+0.5, str(k_val), ha='center', va='center',
                       fontsize=12, color='white', fontweight='bold')
    
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
        if not show:
            plt.close()
        return filename
    
    if show:
        plt.show()
    
    return None


def plot_energy_history(
    energy_history: np.ndarray,
    filename: Optional[str] = None,
    show: bool = False,
    metadata: Optional[Dict] = None
) -> Optional[str]:
    """
    Plot energy evolution during MCMC.
    
    Args:
        energy_history: Array of energy values
        filename: Optional path to save the figure
        show: Whether to display the plot
        metadata: Optional dict with run parameters
        
    Returns:
        Filename if saved, None otherwise
    """
    plt.figure(figsize=(12, 7))
    
    x = np.arange(len(energy_history))
    plt.plot(x, energy_history, linewidth=1.2, color='blue', alpha=0.8)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Energy', fontsize=12)
    
    # Title with metadata
    title = 'Energy Evolution During MCMC'
    if metadata:
        subtitle_parts = []
        if 'board_size' in metadata:
            subtitle_parts.append(f"N={metadata['board_size']}")
        if 'steps' in metadata:
            subtitle_parts.append(f"Steps={metadata['steps']:,}")
        if 'state_space' in metadata:
            subtitle_parts.append(f"State={metadata['state_space']}")
        if 'cooling' in metadata:
            subtitle_parts.append(f"Cooling={metadata['cooling']}")
        if subtitle_parts:
            title += '\n' + ' | '.join(subtitle_parts)
    
    plt.title(title, fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add statistics text box
    final_energy = energy_history[-1]
    min_energy = np.min(energy_history)
    stats_text = f'Final Energy: {final_energy:.0f}\n'
    stats_text += f'Min Energy: {min_energy:.0f}\n'
    stats_text += f'Initial Energy: {energy_history[0]:.0f}'
    
    if metadata:
        if 'accept_rate' in metadata:
            stats_text += f'\nAccept Rate: {metadata["accept_rate"]:.1%}'
        if 'beta_min' in metadata and 'beta_max' in metadata:
            stats_text += f'\nβ: {metadata["beta_min"]} → {metadata["beta_max"]}'
    
    plt.gca().text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        if not show:
            plt.close()
        return filename
    
    if show:
        plt.show()
    
    return None


def plot_averaged_energy_history(
    histories: List[np.ndarray],
    filename: Optional[str] = None,
    show: bool = False,
    metadata: Optional[Dict] = None
) -> Optional[str]:
    """
    Plot averaged energy evolution across multiple runs.
    
    Args:
        histories: List of energy history arrays
        filename: Optional path to save the figure
        show: Whether to display the plot
        metadata: Optional dict with run parameters
        
    Returns:
        Filename if saved, None otherwise
    """
    # Pad histories to same length
    max_len = max(len(h) for h in histories)
    padded = []
    for h in histories:
        if len(h) < max_len:
            padded.append(np.pad(h, (0, max_len - len(h)), mode='edge'))
        else:
            padded.append(h)
    
    histories_arr = np.array(padded)
    mean_energy = np.mean(histories_arr, axis=0)
    std_energy = np.std(histories_arr, axis=0)
    
    plt.figure(figsize=(12, 8))
    x = np.arange(len(mean_energy))
    
    # Plot individual runs in background
    num_runs = len(histories)
    colors = plt.cm.jet(np.linspace(0, 1, num_runs))
    
    for i, h in enumerate(padded):
        plt.plot(h, color=colors[i], alpha=0.3, linewidth=0.8)
    
    # Dummy for legend
    plt.plot([], [], color='gray', alpha=0.5, linewidth=0.8, 
             label=f'Individual Runs ({num_runs})')
    
    # Plot average on top
    plt.plot(x, mean_energy, label='Average Energy', color='black', 
             linewidth=2, zorder=10)
    plt.fill_between(x, mean_energy - std_energy, mean_energy + std_energy,
                     color='black', alpha=0.15, label='±1 Std Dev', zorder=5)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Energy', fontsize=12)
    
    # Title
    title = f'Averaged Energy Evolution ({num_runs} runs)'
    if metadata:
        subtitle_parts = []
        if 'board_size' in metadata:
            subtitle_parts.append(f"N={metadata['board_size']}")
        if 'steps' in metadata:
            subtitle_parts.append(f"Steps={metadata['steps']:,}")
        if 'state_space' in metadata:
            subtitle_parts.append(f"State={metadata['state_space']}")
        if subtitle_parts:
            title += '\n' + ' | '.join(subtitle_parts)
    
    plt.title(title, fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=10)
    
    # Statistics text box
    final_energies = [h[-1] for h in histories]
    stats_text = f'Runs: {num_runs}\n'
    stats_text += f'Avg Final: {np.mean(final_energies):.2f}\n'
    stats_text += f'Best Final: {np.min(final_energies):.0f}\n'
    stats_text += f'Worst Final: {np.max(final_energies):.0f}'
    
    if metadata:
        if 'cooling' in metadata:
            stats_text += f'\nCooling: {metadata["cooling"]}'
        if 'beta_min' in metadata and 'beta_max' in metadata:
            stats_text += f'\nβ: {metadata["beta_min"]} → {metadata["beta_max"]}'
    
    plt.gca().text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                   fontsize=10, verticalalignment='top',
                   fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        if not show:
            plt.close()
        return filename
    
    if show:
        plt.show()
    
    return None


def save_results(
    output_dir: str,
    board,
    energy_history: np.ndarray,
    metadata: Dict,
    save_plots: bool = True,
    save_data: bool = True
) -> Dict[str, str]:
    """
    Save all results to a directory with proper naming.
    
    Args:
        output_dir: Directory to save results
        board: Final board state
        energy_history: Energy history array
        metadata: Dict with all run parameters
        save_plots: Whether to save visualization plots
        save_data: Whether to save energy data as JSON/NPY
        
    Returns:
        Dict mapping result type to filename
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create filename prefix from parameters
    N = metadata.get('board_size', board.N)
    state_space = metadata.get('state_space', 'unknown')
    steps = metadata.get('steps', len(energy_history))
    seed = metadata.get('seed', 0)
    cooling = metadata.get('cooling', 'unknown')
    
    prefix = f"N{N}_{state_space}_{cooling}_steps{steps}_seed{seed}"
    
    saved_files = {}
    
    if save_plots:
        # Save 3D solution visualization
        sol_file = output_path / f"{prefix}_solution.png"
        visualize_solution(board, filename=str(sol_file), metadata=metadata)
        saved_files['solution'] = str(sol_file)
        
        # Save Latin square visualization
        latin_file = output_path / f"{prefix}_latin_square.png"
        visualize_latin_square(board, filename=str(latin_file), metadata=metadata)
        saved_files['latin_square'] = str(latin_file)
        
        # Save energy history plot
        energy_file = output_path / f"{prefix}_energy_history.png"
        plot_energy_history(energy_history, filename=str(energy_file), metadata=metadata)
        saved_files['energy_plot'] = str(energy_file)
    
    if save_data:
        # Save energy history as numpy
        npy_file = output_path / f"{prefix}_energy_history.npy"
        np.save(str(npy_file), energy_history)
        saved_files['energy_npy'] = str(npy_file)
        
        # Save metadata as JSON
        json_file = output_path / f"{prefix}_metadata.json"
        
        # Convert non-serializable types
        json_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, np.ndarray):
                json_metadata[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                json_metadata[k] = float(v)
            else:
                json_metadata[k] = v
        
        json_metadata['final_energy'] = float(board.energy)
        json_metadata['timestamp'] = datetime.now().isoformat()
        
        with open(json_file, 'w') as f:
            json.dump(json_metadata, f, indent=2)
        saved_files['metadata'] = str(json_file)
    
    return saved_files


def save_competition_format(
    board,
    filename: str
) -> str:
    """
    Save solution in competition format.
    
    Format: N² rows, each containing 3 integers x,y,z from 0 to N-1,
    separated by commas, where (x,y,z) denotes the position of a queen.
    
    No headers, no comments - just the positions.
    
    Args:
        board: Board state with get_queens() method
        filename: Path to save the file
        
    Returns:
        Path to saved file
    """
    queens = np.array(board.get_queens())
    
    with open(filename, 'w') as f:
        # Write queen positions only - no headers
        for q in queens:
            f.write(f"{int(q[0])},{int(q[1])},{int(q[2])}\n")
    
    return filename


def create_run_output_folder(
    base_output_dir: str,
    board_size: int,
    seed: int
) -> str:
    """
    Create a timestamped output folder for a run.
    
    Structure: base_output_dir/N{board_size}/run_{datetime}_{seed}/
    
    Args:
        base_output_dir: Base output directory
        board_size: Board dimension N
        seed: Random seed used
        
    Returns:
        Path to created folder
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = Path(base_output_dir) / f"N{board_size}" / f"run_{timestamp}_seed{seed}"
    run_folder.mkdir(parents=True, exist_ok=True)
    return str(run_folder)


def save_run_results(
    output_dir: str,
    board,
    energy_history: np.ndarray,
    metadata: Dict,
    save_plots: bool = True,
    save_data: bool = True
) -> Dict[str, str]:
    """
    Save all results for a single run to a timestamped folder.
    
    Creates: output_dir/N{size}/run_{datetime}_seed{seed}/
    
    Always saves:
    - solution.txt: Competition format (x,y,z per line)
    - metadata.json: Run parameters and results
    
    Optionally saves:
    - solution.png: 3D visualization
    - latin_square.png: Latin square projection
    - energy_history.png: Energy plot
    - energy_history.npy: Raw energy data
    
    Args:
        output_dir: Base output directory
        board: Final board state
        energy_history: Energy history array
        metadata: Dict with all run parameters
        save_plots: Whether to save visualization plots
        save_data: Whether to save energy data as NPY
        
    Returns:
        Dict mapping result type to filename
    """
    N = metadata.get('board_size', board.N)
    seed = metadata.get('seed', 0)
    
    # Create timestamped run folder
    run_folder = create_run_output_folder(output_dir, N, seed)
    run_path = Path(run_folder)
    
    saved_files = {'run_folder': run_folder}
    
    # Always save competition format solution (positions only, no headers)
    solution_txt = run_path / "solution.txt"
    save_competition_format(board, str(solution_txt))
    saved_files['solution_txt'] = str(solution_txt)
    
    # Always save metadata JSON
    json_file = run_path / "metadata.json"
    json_metadata = {}
    for k, v in metadata.items():
        if isinstance(v, np.ndarray):
            json_metadata[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            json_metadata[k] = float(v)
        else:
            json_metadata[k] = v
    
    # Compute both hash and iter energy
    queens = board.get_queens()
    queens_list = [(int(q[0]), int(q[1]), int(q[2])) for q in queens]
    
    # Iter energy (true energy - count attacking pairs)
    # Also track which queens are endangered (under attack)
    iter_energy = 0
    n = len(queens_list)
    endangered_queens = set()
    for i in range(n):
        for j in range(i + 1, n):
            if check_attack(queens_list[i], queens_list[j]):
                iter_energy += 1
                endangered_queens.add(i)
                endangered_queens.add(j)
    
    # Endangered energy (number of queens under attack)
    endangered_energy = len(endangered_queens)
    
    # Hash energy (surrogate energy from line counts)
    # Always compute from line counts to get the true hash energy
    line_counts = board.get_line_counts()
    hash_energy = 0.0
    for key in line_counts:
        counts = line_counts[key]
        # Energy contribution: C(n,2) = n*(n-1)/2 for each line
        pairs = (counts * (counts - 1)) / 2
        hash_energy += float(np.sum(pairs))
    
    # Colored energy (conflicts + 4 * black squares)
    black_count = sum(1 for q in queens_list if (q[0] + q[1] + q[2]) % 2 == 1)
    colored_energy = iter_energy + 4 * black_count
    
    # Weighted energy (conflicts + sum|x+y-2z|)
    total_weight = sum(abs(q[0] + q[1] - 2 * q[2]) for q in queens_list)
    weighted_energy = iter_energy + total_weight
    
    # Colored endangered energy (endangered + 4 * black squares)
    colored_endangered_energy = endangered_energy + 4 * black_count
    
    # Weighted endangered energy (endangered + sum|x+y-2z|)
    weighted_endangered_energy = endangered_energy + total_weight
    
    json_metadata['final_energy_hash'] = hash_energy
    json_metadata['final_energy_iter'] = iter_energy
    json_metadata['final_energy_endangered'] = endangered_energy
    json_metadata['final_energy_colored'] = colored_energy
    json_metadata['final_energy_weighted'] = weighted_energy
    json_metadata['final_energy_colored_endangered'] = colored_endangered_energy
    json_metadata['final_energy_weighted_endangered'] = weighted_endangered_energy
    json_metadata['black_squares_count'] = black_count
    json_metadata['total_weight'] = total_weight
    json_metadata['complexity'] = metadata.get('complexity', 'unknown')
    json_metadata['energy_treatment'] = metadata.get('energy_treatment', 'linear')
    json_metadata['timestamp'] = datetime.now().isoformat()
    json_metadata['queens'] = [[int(q[0]), int(q[1]), int(q[2])] for q in queens]
    
    with open(json_file, 'w') as f:
        json.dump(json_metadata, f, indent=2)
    saved_files['metadata'] = str(json_file)
    
    if save_plots:
        # Save 3D solution visualization
        sol_file = run_path / "solution.png"
        visualize_solution(board, filename=str(sol_file), metadata=metadata)
        saved_files['solution_png'] = str(sol_file)
        
        # Save Latin square visualization
        latin_file = run_path / "latin_square.png"
        visualize_latin_square(board, filename=str(latin_file), metadata=metadata)
        saved_files['latin_square'] = str(latin_file)
        
        # Save energy history plot
        energy_file = run_path / "energy_history.png"
        plot_energy_history(energy_history, filename=str(energy_file), metadata=metadata)
        saved_files['energy_plot'] = str(energy_file)
    
    if save_data:
        # Save energy history as numpy
        npy_file = run_path / "energy_history.npy"
        np.save(str(npy_file), energy_history)
        saved_files['energy_npy'] = str(npy_file)
    
    return saved_files


def save_multiple_results(
    output_dir: str,
    results: List[Tuple],
    histories: List[np.ndarray],
    metadata: Dict,
    save_plots: bool = True,
    save_data: bool = True
) -> Dict[str, str]:
    """
    Save results from multiple runs.
    
    Args:
        output_dir: Directory to save results
        results: List of (board, history, accept_rate) tuples
        histories: List of energy history arrays
        metadata: Dict with all run parameters
        save_plots: Whether to save visualization plots
        save_data: Whether to save energy data
        
    Returns:
        Dict mapping result type to filename
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    N = metadata.get('board_size', results[0][0].N)
    state_space = metadata.get('state_space', 'unknown')
    steps = metadata.get('steps', 0)
    cooling = metadata.get('cooling', 'unknown')
    num_runs = len(results)
    
    prefix = f"N{N}_{state_space}_{cooling}_steps{steps}_runs{num_runs}"
    
    saved_files = {}
    
    if save_plots:
        # Save averaged energy history
        avg_file = output_path / f"{prefix}_averaged_energy.png"
        plot_averaged_energy_history(histories, filename=str(avg_file), metadata=metadata)
        saved_files['averaged_energy'] = str(avg_file)
        
        # Save best solution
        final_energies = [r[0].energy for r in results]
        best_idx = np.argmin(final_energies)
        best_board = results[best_idx][0]
        
        best_file = output_path / f"{prefix}_best_solution.png"
        visualize_solution(best_board, filename=str(best_file), metadata=metadata)
        saved_files['best_solution'] = str(best_file)
    
    if save_data:
        # Save all histories
        all_histories = np.array([h for h in histories])
        npy_file = output_path / f"{prefix}_all_histories.npy"
        np.save(str(npy_file), all_histories)
        saved_files['histories_npy'] = str(npy_file)
        
        # Save summary JSON
        json_file = output_path / f"{prefix}_summary.json"
        
        final_energies = [float(r[0].energy) for r in results]
        accept_rates = [float(r[2]) for r in results]
        
        summary = {
            **{k: v for k, v in metadata.items() if not isinstance(v, np.ndarray)},
            'num_runs': num_runs,
            'final_energies': final_energies,
            'accept_rates': accept_rates,
            'avg_final_energy': float(np.mean(final_energies)),
            'best_final_energy': float(np.min(final_energies)),
            'worst_final_energy': float(np.max(final_energies)),
            'success_rate': float(sum(e == 0 for e in final_energies) / num_runs),
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)
        saved_files['summary'] = str(json_file)
    
    return saved_files
