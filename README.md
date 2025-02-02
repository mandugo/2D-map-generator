# 2D Map Generation

This repository contains a Python script for generating different types of 2D grid maps. Each map is represented as an (NxN) matrix, where each cell can be:

- **0** – a free cell (walkable area),
- **1** – an obstacle (non-walkable).

One key feature is that all generated maps **guarantee at least one valid path** from the top-left cell `(0,0)` to the bottom-right cell `(N-1,N-1)`.

---

## Contents

1. [Features](#features)   
2. [Usage](#usage)  
3. [Parameters](#parameters)  
   - [Random Map](#random-map)  
   - [City-Block Map](#city-block-map)  
   - [Maze with Loops](#maze-with-loops)  
4. [Functions Overview](#functions-overview)  
5. [Example](#example)  
6. [License](#license)

---

## Features

- **Multiple Map Types**  
  - **Random** obstacles with a specified probability.  
  - **City-Block** layout, simulating “roads” and “blocks.”  
  - **Maze** with extra passages, ensuring multiple routes from start to end.

- **Guaranteed Path**  
  The script uses a BFS check to verify that each generated map is traversable from \((0,0)\) to \((N-1,N-1)\). If not, it regenerates or raises an error after a specified number of attempts.

- **Easy Visualization**  
  A helper function (`plot_grid`) utilizes Matplotlib to plot the final map, showing free cells as white and obstacles as black.

- **Customizable**  
  Users can tweak various parameters (obstacle probability, city block size, maze loop factor, etc.) to generate diverse layouts.

---

## Usage

Install Dependencies:
```
pip install matplotlib numpy
```

Run the main script:
```
python map_generator.py
```

Depending on how you set your parameters, you can generate and visualize:
- A random map.
- A city-block map.
- A maze with multiple paths.

Within the script, adjust parameters like `N`, `map_type`, `obstacle_prob`, etc.

---

## Parameters

Below are the main parameters you can tweak when calling `generate_map`:

### Random Map
- `map_type='random'`
- `obstacle_prob (float)` – Probability of placing an obstacle in a free cell.

Example: 
```
my_random_map = generate_map(N=15, map_type='random', obstacle_prob=0.2)
```

### City-Block Map
- `map_type='city'`
- `block_size (int)` – The size of each “building block.”
- `obstacle_fill (float)` – Within a block, the probability that a given cell is an obstacle.

Example: 
```
my_city_map = generate_map(N=15, map_type='city', block_size=3, obstacle_fill=0.7)
```

### Maze with Loops
- `map_type='maze'`
- `extra_paths_prob (float)` – After building a “perfect” maze (exactly one path), the script randomly removes additional walls with this probability. This adds loops and multiple routes.

Example: 
```
my_maze = generate_map(N=15, map_type='maze', extra_paths_prob=0.1)
```

---

## Functions Overview
```
generate_map(N, map_type='random', obstacle_prob=0.2, block_size=3,
            obstacle_fill=0.7, extra_paths_prob=0.05, max_tries=100)
```

Main entry point to generate the grid.
- **Parameters**
   - `N (int)`: Size of the grid, (NxN).
   - `map_type (str)`: `'random'`, `'city'`, or `'maze'`.
   - `obstacle_prob (float)`: Probability of placing an obstacle (used only if `map_type='random'`).
   - `block_size (int)`: Size of each city block (used only if `map_type='city'`).
   - `obstacle_fill (float)`: Probability of a cell in a block being an obstacle (used only if `map_type='city'`).
   - `extra_paths_prob (float)`: Probability of removing additional walls to form loops (used only if `map_type='maze'`).
   - `max_tries (int)`: How many times to attempt generating a valid map before raising an error.
- **Returns**
   - A 2D list (`list[list[int]]`) where 0 = free cell, 1 = obstacle.

```
plot_grid(grid, title="2D Map")
```
Visualizes the final grid using Matplotlib, with black squares for obstacles and white squares for free cells.

```
get_obstacle_coordinates(grid, one_based=True)
```
Returns a list of `(row, col)` positions of all obstacles.
- If one_based=True, positions range from `(1,1)` to `(N,N)`.
- If False, positions range from `(0,0)` to `(N-1,N-1)`.

```
is_reachable(grid)
```
Internal helper that uses BFS to check if the top-left cell `(0,0)` can reach the bottom-right cell `(N-1,N-1)`.

---

## Example
```
# Clone the repo and install dependencies (matplotlib, numpy)
git clone https://github.com/mandugo/2D-map-generator.git
cd 2D-map-generator
pip install matplotlib numpy

# Run the main script
python map_generator.py
```
Sample Output:
1. A Random Map (`obstacle_prob=0.3`)
2. A City-Block Map with `block_size=3` and `obstacle_fill=0.6`
3. A Maze with loops (`extra_paths_prob=0.1`)

Each map will pop up in a Matplotlib window, and the script will also print obstacle coordinates for the random map in the console.

---

## License
This project is licensed under the MIT License – see the LICENSE file for details.
Feel free to use and modify this code in your own projects. Contributions and improvements are welcome!
