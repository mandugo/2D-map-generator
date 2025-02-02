import random
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def is_reachable(grid):
    """
    Check if there's a path from the top-left (0,0) to bottom-right (N-1, N-1)
    using BFS. Returns True if reachable, False otherwise.
    """
    N = len(grid)
    # If start or end is blocked, no path is possible.
    if grid[0][0] == 1 or grid[N-1][N-1] == 1:
        return False

    visited = [[False]*N for _ in range(N)]
    visited[0][0] = True
    queue = deque([(0, 0)])
    
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    
    while queue:
        r, c = queue.popleft()
        if (r, c) == (N-1, N-1):
            return True
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < N and 0 <= nc < N:
                if not visited[nr][nc] and grid[nr][nc] == 0:
                    visited[nr][nc] = True
                    queue.append((nr, nc))
    
    return False

def generate_random_map(N, obstacle_prob=0.2, max_tries=100):
    """
    Generates an N x N grid with random obstacles (1) and free cells (0).
    Ensures at least one path from (0,0) to (N-1, N-1).
    """
    for _ in range(max_tries):
        grid = [[0]*N for _ in range(N)]
        
        # Randomly place obstacles except at start and end
        for r in range(N):
            for c in range(N):
                if (r, c) not in [(0,0), (N-1,N-1)]:
                    if random.random() < obstacle_prob:
                        grid[r][c] = 1
        
        # Check connectivity
        if is_reachable(grid):
            return grid
    
    raise RuntimeError("Could not generate a valid random map within max_tries.")

def generate_city_map(N, block_size=3, obstacle_fill=0.7, max_tries=100):
    """
    Generates an N x N grid in a city-block style:
    - 'Blocks' are mostly obstacles
    - Single-cell roads separate blocks
    Ensures at least one path from (0,0) to (N-1, N-1).
    """
    for _ in range(max_tries):
        grid = [[0]*N for _ in range(N)]
        
        for r in range(N):
            for c in range(N):
                # Determine if we're on a 'road' (every block_size+1)
                if (r % (block_size+1) == block_size) or (c % (block_size+1) == block_size):
                    grid[r][c] = 0  # road
                else:
                    # inside a block
                    if random.random() < obstacle_fill:
                        grid[r][c] = 1
        
        # Ensure start and end cells are free
        grid[0][0] = 0
        grid[N-1][N-1] = 0
        
        # Check connectivity
        if is_reachable(grid):
            return grid
    
    raise RuntimeError("Could not generate a valid city map within max_tries.")

def generate_perfect_maze(N):
    """
    Creates a 'perfect' maze using DFS (backtracking).
    A 'perfect' maze has exactly one path between any two cells.
    0 = free, 1 = wall.
    """
    maze = [[1]*N for _ in range(N)]
    
    start_r, start_c = 0, 0
    maze[start_r][start_c] = 0
    
    stack = [(start_r, start_c)]
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    
    while stack:
        r, c = stack[-1]
        random.shuffle(directions)
        carved = False
        
        for dr, dc in directions:
            nr, nc = r + 2*dr, c + 2*dc
            if 0 <= nr < N and 0 <= nc < N and maze[nr][nc] == 1:
                # Check surroundings to avoid loops in perfect maze
                free_count = 0
                for ddr, ddc in directions:
                    rr, cc = nr + ddr, nc + ddc
                    if 0 <= rr < N and 0 <= cc < N and maze[rr][cc] == 0:
                        free_count += 1
                if free_count <= 1:
                    maze[r+dr][c+dc] = 0
                    maze[nr][nc] = 0
                    stack.append((nr, nc))
                    carved = True
                    break
        
        if not carved:
            stack.pop()
    
    # Force the end to be free
    maze[N-1][N-1] = 0
    return maze

def generate_maze_with_loops(N, extra_paths_prob=0.05, max_tries=100):
    """
    Generates a maze with multiple paths by:
      1) Creating a perfect maze.
      2) Randomly removing walls with probability extra_paths_prob to add loops.
    Ensures at least one path from (0,0) to (N-1,N-1).
    """
    for _ in range(max_tries):
        maze = generate_perfect_maze(N)
        
        # Randomly remove walls to create loops
        for r in range(N):
            for c in range(N):
                if maze[r][c] == 1 and random.random() < extra_paths_prob:
                    maze[r][c] = 0
        
        # Ensure start and end are free
        maze[0][0] = 0
        maze[N-1][N-1] = 0
        
        if is_reachable(maze):
            return maze
    
    raise RuntimeError("Could not generate a valid maze-with-loops within max_tries.")

def generate_map(N, 
                 map_type='random', 
                 obstacle_prob=0.2, 
                 block_size=3, 
                 obstacle_fill=0.7,
                 extra_paths_prob=0.05,
                 max_tries=100):
    """
    Main function to generate a map of size N x N.
    
    Parameters
    ----------
    N : int
        Grid size (N x N).
    map_type : str, optional
        One of 'random', 'city', or 'maze'.
    obstacle_prob : float, optional
        Used only if map_type='random'. Probability of an obstacle in each cell.
    block_size : int, optional
        Used only if map_type='city'. Size of each city block.
    obstacle_fill : float, optional
        Used only if map_type='city'. Probability of filling a block cell with an obstacle.
    extra_paths_prob : float, optional
        Used only if map_type='maze'. Probability of removing extra walls to create loops.
    max_tries : int, optional
        Max attempts to generate a valid map.
    
    Returns
    -------
    grid : list of lists (int)
        2D grid where 0 = free, 1 = obstacle.
    """
    if map_type == 'random':
        return generate_random_map(N, obstacle_prob=obstacle_prob, max_tries=max_tries)
    elif map_type == 'city':
        return generate_city_map(N, block_size=block_size, obstacle_fill=obstacle_fill, max_tries=max_tries)
    elif map_type == 'maze':
        return generate_maze_with_loops(N, extra_paths_prob=extra_paths_prob, max_tries=max_tries)
    else:
        raise ValueError(f"Unknown map_type '{map_type}'. Choose from 'random', 'city', or 'maze'.")

def plot_grid(grid, title="2D Map", show_start_end=True, grid_lines=False):
    """
    Visualize the grid with Matplotlib:
      - 0 = free
      - 1 = obstacle

    Possible improvements:
      - Use a discrete colormap that forces 0 and 1 to be different colors.
      - Force the color range from 0 to 1 so we don't get "squashed" colorbars.
      - Optionally show start and end cells in a special color/marker.
      - Optionally draw grid lines for clarity.
    """
    data = np.array(grid)
    N = data.shape[0]

    # Create a discrete colormap for 0 and 1.
    # White for free cells, black for obstacles.
    # (You could also invert it or pick different color combos.)
    cmap = ListedColormap(["white", "black"])

    # Plot with a fixed vmin/vmax so 0=white and 1=black
    plt.figure(figsize=(6,6))
    plt.imshow(data, cmap=cmap, origin='upper', vmin=0, vmax=1)
    
    # Optional: Add a grid of lines between cells
    if grid_lines:
        # Turn on minor ticks for grid lines
        plt.grid(True, which='minor', color='gray', linewidth=0.5)
        plt.minorticks_on()
        plt.xticks(np.arange(-0.5, N, 1))
        plt.yticks(np.arange(-0.5, N, 1))
        plt.xlim(-0.5, N - 0.5)
        plt.ylim(N - 0.5, -0.5)  # because origin='upper', y goes downward

    # If desired, highlight start and end cells
    if show_start_end:
        plt.scatter(x=[0], y=[0], s=100, c='red', edgecolors='white', marker='o', label='Start')
        plt.scatter(x=[N-1], y=[N-1], s=100, c='green', edgecolors='white', marker='X', label='Goal')
        plt.legend(loc='upper right')

    plt.title(title)
    plt.tight_layout()
    plt.show()

def get_obstacle_coordinates(grid, one_based=True):
    """
    Returns a list of obstacle coordinates where grid[row][col] == 1.
    
    If one_based=True, coordinates are in 1-based indexing (1..N).
    Otherwise, 0-based indexing (0..N-1).
    """
    obstacles = []
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 1:
                if one_based:
                    obstacles.append((r+1, c+1))
                else:
                    obstacles.append((r, c))
    return obstacles

if __name__ == "__main__":
    # Example usage
    N = 10
    max_tries = 200
    
    # 1) Random map
    obstacle_prob = 0.3
    random_grid = generate_map(N, map_type='random', obstacle_prob=obstacle_prob, max_tries=max_tries)
    plot_grid(random_grid, title=f"Random Map (N={N}, Obstacles ~ {int(obstacle_prob*100)}%)")
    
    # 2) City-block map
    block_size = 4
    obstacle_fill = 0.8
    city_grid = generate_map(N, map_type='city', block_size=block_size, obstacle_fill=obstacle_fill, max_tries=max_tries)
    plot_grid(city_grid, title=f"City-Block Map (N={N}, block_size={block_size}, fill={obstacle_fill})")
    
    # 3) Maze with multiple paths
    extra_paths_prob = 0.2
    maze_grid = generate_map(N, map_type='maze', extra_paths_prob=extra_paths_prob, max_tries=max_tries)
    plot_grid(maze_grid, title=f"Maze with Multiple Paths (N={N}, extra_paths_prob={extra_paths_prob})")
    
    # Print obstacle coordinates in 1-based indexing for the random map
    obs_coords = get_obstacle_coordinates(random_grid, one_based=True)
    #print("Obstacle coordinates (1-based) for the Random Map:")
    #print(obs_coords)