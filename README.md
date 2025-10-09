# Hexagonal Maze Treasure Hunt Solver

This Python program implements a solver for a hexagonal maze treasure hunt using the A* search algorithm. The maze includes treasures to collect, traps with various effects, rewards that reduce movement costs, and obstacles that block paths. The program provides an interactive visualization of the maze and the computed optimal path using Matplotlib.

## Maze Overview

![Maze Preview](https://github.com/samuellee-e/Hexagonal-Maze-Treasure-Hunt-Solver/blob/main/images/maze.png)


## Features
- **Hexagonal Grid Navigation**: Implements proper neighbor calculations for a hexagonal grid layout.
- **A* Pathfinding**: Finds the optimal path to collect all treasures, supporting multiple optimization modes:
  - `steps`: Minimizes the total number of steps.
  - `energy`: Minimizes the total energy consumed.
  - `combined`: Balances steps and energy for an overall optimal path.
- **Traps and Rewards**:
  - **Trap1**: Doubles energy cost for future moves.
  - **Trap2**: Doubles step cost for future moves.
  - **Trap3**: Teleports two steps in the last movement direction.
  - **Trap4**: Invalidates the entire path (instant failure).
  - **Reward1**: Halves energy cost for future moves.
  - **Reward2**: Halves step cost for future moves.
- **Visualization**: Displays the maze with special cells (treasures, traps, rewards, obstacles) and highlights the solution path with start/end markers and movement arrows.
- **Energy Management**: Tracks remaining energy, starting with 50 units, with moves consuming energy based on active effects.

## Requirements
- Python 3.x
- Required libraries:
  - `matplotlib.pyplot` (for plotting the maze visualization)
  - `matplotlib.patches` (for rendering hexagonal shapes)
  - `math` (for geometric calculations in hexagonal grid positioning)
  - `heapq` (for priority queue in A* algorithm)
  - `itertools.permutations` (for generating treasure collection orders in heuristic)
- Install dependencies using:
  ```bash
  pip install matplotlib
  ```

## Usage
1. **Run the Program**:
   ```bash
   python hexagonal_maze_solver.py
   ```
2. **Program Flow**:
   - The program displays the initial maze layout.
   - The user is prompted to manually input the optimization mode (`steps`, `energy`, or `combined`) via the console.
   - The A* algorithm computes the optimal path from position (0, 0) to collect all treasures.
   - If a solution is found, it displays:
     - Path details (steps, energy, treasures collected, active effects).
     - A visualization of the path with start (green 'S'), end (blue 'E'), and movement arrows (red).
   - If no solution is found, it explains possible reasons and redisplays the original maze.

## File Structure
- `hexagonal_maze_solver.py`: Main Python script containing the solver and visualization logic.
- **Maze Configuration** (defined in the script):
  - Treasures: Located at `(3,4), (4,1), (7,3), (9,3)`.
  - Traps: Four types with specific effects at predefined locations.
  - Rewards: Two types that reduce movement costs.
  - Obstacles: Impassable cells at specified positions.
  - Grid Size: 10 rows x 6 columns.

## How It Works
1. **Maze Setup**: The maze is represented as a dictionary mapping `(row, col)` coordinates to cell information (content and valid moves).
2. **A* Algorithm**:
   - Uses a `State` class to track position, path, steps, energy, collected treasures, active effects, and remaining energy.
   - Implements a heuristic function to estimate the cost to collect remaining treasures, considering different visit orders.
   - Handles trap and reward effects, including teleportation for Trap3.
   - Avoids paths with insufficient energy or those hitting Trap4.
3. **Visualization**:
   - Uses Matplotlib to render a hexagonal grid with proper offsets.
   - Colors and symbols differentiate cell types (e.g., orange for treasures, purple for traps).
   - Displays the solution path with arrows and start/end markers.
   - Includes a legend explaining cell types.

## Example Output
Upon running, the program:
1. Shows the initial maze with treasures, traps, rewards, and obstacles.
2. Prompts the user to manually enter an optimization mode (e.g., `combined`) in the console.
3. Outputs search progress and results, such as:
   ```
   ==================== HEXAGONAL MAZE TREASURE HUNT SOLVER ====================
   Using A* Algorithm with Visualization

   [+] Displaying original map...
   [+] Maze Information
     - Total treasures      : 4
     - Total traps          : 6
     - Total rewards        : 4
     - Total obstacles      : 9
   [+] Search Configuration
   Enter optimization mode (steps/energy/combined): combined
   [+] Starting A* search from (0, 0) with mode 'combined'
   [SUCCESS] Found solution after 1234 iterations
   [+] Solution Found!
     - Total steps taken    : 15
     - Total energy used    : 12
     - Remaining energy     : 38
     - Path length          : 16
     - Treasures collected  : 4
     - Treasure positions   : [(3, 4), (4, 1), (7, 3), (9, 3)]
     - Active effects       : ['half_energy']
   Path:
     (0,0) → (1,0) → (1,1) → (2,1) → (3,1)
     ...
   [+] Displaying solution path...
   ```
4. Shows the final visualization with the optimal path highlighted.

## Function Descriptions
Below is a detailed explanation of each function in the program, covering their purpose, parameters, return values, and key logic.

### `get_neighbors(rows, cols, pos, obstacles)`
- **Purpose**: Calculates valid neighboring positions for a given position in the hexagonal grid, accounting for grid boundaries and obstacles.
- **Parameters**:
  - `rows` (int): Number of rows in the maze (e.g., 10).
  - `cols` (int): Number of columns in the maze (e.g., 6).
  - `pos` (tuple): Current position as `(row, col)`.
  - `obstacles` (set): Set of `(row, col)` positions that are impassable.
- **Returns**: List of valid neighboring `(row, col)` positions.
- **Logic**:
  - Selects the direction mapping (`HEX_DIRECTIONS_EVEN` or `HEX_DIRECTIONS_ODD`) based on whether the row is even or odd, as hexagonal grids have different neighbor offsets.
  - Iterates through the six possible directions, computing new positions.
  - Filters out positions that are out of bounds or occupied by obstacles.
  - Used by `generate_maze_map` and `get_successors` to determine valid moves.

### `generate_maze_map()`
- **Purpose**: Creates a dictionary representing the maze, mapping each cell to its content and valid moves.
- **Parameters**: None.
- **Returns**: Dictionary mapping `(row, col)` to a dictionary with keys `content` (e.g., 'Treasure', 'Trap1') and `moves` (list of valid neighbors).
- **Logic**:
  - Iterates over a 10x6 grid, assigning each cell a default content of 'Empty'.
  - Updates content based on predefined `TREASURES`, `TRAPS`, `REWARDS`, and `OBSTACLES`.
  - Computes valid moves for each cell using `get_neighbors`.
  - Serves as the core data structure for the A* algorithm.

### `apply_direction(pos, direction_name)`
- **Purpose**: Computes the new position after moving in a specified direction, used primarily for Trap3 teleportation.
- **Parameters**:
  - `pos` (tuple): Starting `(row, col)` position.
  - `direction_name` (str): Direction (e.g., 'up', 'bottom-left').
- **Returns**: New `(row, col)` position after applying the direction.
- **Logic**:
  - Uses the appropriate direction mapping based on row parity.
  - Adds the direction’s row and column offsets to the current position.
  - Called by `get_successors` to handle Trap3’s two-step teleportation.

### `find_direction_name(r0, c0, r1, c1)`
- **Purpose**: Determines the direction name for moving from one position to another, used to track movement direction for Trap3.
- **Parameters**:
  - `r0, c0` (int): Starting row and column.
  - `r1, c1` (int): Ending row and column.
- **Returns**: Direction name (str) or `None` if the move is invalid.
- **Logic**:
  - Uses the direction mapping for the starting row’s parity.
  - Checks each direction to find one that matches the move from `(r0, c0)` to `(r1, c1)`.
  - Returns the matching direction name or `None` if no valid direction is found.

### `get_successors(state, maze_map)`
- **Purpose**: Generates all valid successor states from the current state, handling movement, trap/reward effects, and energy constraints.
- **Parameters**:
  - `state` (State): Current state object with position, path, steps, energy, etc.
  - `maze_map` (dict): Maze data structure from `generate_maze_map`.
- **Returns**: List of `State` objects representing valid next states.
- **Logic**:
  - Iterates over valid neighbors from the maze map.
  - Computes movement direction using `find_direction_name`.
  - Calculates step and energy costs based on active effects (via `step_multiplier` and `energy_multiplier`).
  - Handles special cases:
    - **Trap3**: Teleports two steps in the movement direction, updating position and path if valid.
    - **Trap4**: Returns an empty list, invalidating the path.
    - **Trap1/Trap2**: Adds effects to double future energy/step costs.
    - **Reward1/Reward2**: Adds effects to halve future energy/step costs.
    - **Treasure**: Adds the position to collected treasures.
  - Checks remaining energy to ensure moves are feasible.
  - Creates new `State` objects for each valid successor.
  - Core function for A* state expansion.

### `hex_distance(a, b)`
- **Purpose**: Calculates the shortest distance between two points in a hexagonal grid using cube coordinates.
- **Parameters**:
  - `a, b` (tuple): `(row, col)` positions to compare.
- **Returns**: Integer representing the minimum number of steps between positions.
- **Logic**:
  - Converts offset coordinates to cube coordinates (x, y, z) where x + y + z = 0.
  - Computes the maximum absolute difference between cube coordinates, which gives the hexagonal distance.
  - Used in the `heuristic` function to estimate distances to treasures.

### `step_multiplier(effects)`
- **Purpose**: Computes the step cost multiplier based on active effects.
- **Parameters**:
  - `effects` (set): Set of active effect names (e.g., 'double_step', 'half_step').
- **Returns**: Float multiplier (e.g., 2.0 for double, 0.5 for half, 1.0 for no effect).
- **Logic**:
  - Starts with a multiplier of 1.0.
  - Applies a 2x multiplier if 'double_step' is active.
  - Applies a 0.5x multiplier if 'half_step' is active.
  - Used in `get_successors` and `heuristic` to adjust step costs.

### `energy_multiplier(effects)`
- **Purpose**: Computes the energy cost multiplier based on active effects.
- **Parameters**:
  - `effects` (set): Set of active effect names (e.g., 'double_energy', 'half_energy').
- **Returns**: Float multiplier (e.g., 2.0, 0.5, 1.0).
- **Logic**:
  - Similar to `step_multiplier`, but for energy costs.
  - Applies 2x for 'double_energy' and 0.5x for 'half_energy'.
  - Used in `get_successors` and `heuristic` to adjust energy costs.

### `heuristic(state, maze_map, mode='steps')`
- **Purpose**: Estimates the minimum cost to collect all remaining treasures, used in A* to guide the search.
- **Parameters**:
  - `state` (State): Current state with position, treasures collected, and effects.
  - `maze_map` (dict): Maze data structure.
  - `mode` (str): Optimization mode ('steps', 'energy', or 'combined').
- **Returns**: Float representing the estimated cost to complete the goal.
- **Logic**:
  - Identifies uncollected treasures from the maze map.
  - Uses a nested function `estimate_path_cost` to try all permutations of remaining treasures.
  - For each permutation:
    - Computes distances between consecutive positions using `hex_distance`.
    - Applies step/energy multipliers based on current effects.
    - Calculates total cost based on the optimization mode.
  - Returns the minimum cost across all permutations.
  - Balances admissibility and efficiency for A* search.

### `a_star_search(start, maze_map, total_treasures, mode='combined', verbose=False)`
- **Purpose**: Executes the A* search algorithm to find an optimal path collecting all treasures.
- **Parameters**:
  - `start` (State): Initial state (starting at `(0, 0)`).
  - `maze_map` (dict): Maze data structure.
  - `total_treasures` (int): Number of treasures to collect (e.g., 4).
  - `mode` (str): Optimization mode ('steps', 'energy', 'combined').
  - `verbose` (bool): If True, prints search progress.
- **Returns**: `State` object for the goal state if found, `None` otherwise.
- **Logic**:
  - Maintains a priority queue (`frontier`) of states, prioritized by f-cost (g + h).
  - Tracks visited states with a key based on position, collected treasures, effects, and remaining energy.
  - For each state:
    - Checks if all treasures are collected (goal reached).
    - Skips states with equal or better costs in `visited`.
    - Generates successors using `get_successors`.
    - Computes heuristic costs using `heuristic`.
    - Adds successors to the frontier with their f-costs.
  - Prints success/failure and iteration count if verbose.
  - Core algorithm for pathfinding.

### `print_header(text)`
- **Purpose**: Prints a formatted header for program sections in the console output.
- **Parameters**:
  - `text` (str): Header text to display.
- **Returns**: None.
- **Logic**:
  - Prints an 80-character line of '='.
  - Centers the text within an 80-character line.
  - Used for visual organization in the main program output.

### `print_section(text)`
- **Purpose**: Prints a formatted section header in the console output.
- **Parameters**:
  - `text` (str): Section text to display.
- **Returns**: None.
- **Logic**:
  - Prints the text prefixed with '[+]'.
  - Used for subsections in the main program output.

### `print_info(label, value)`
- **Purpose**: Prints formatted information with a label and value in the console output.
- **Parameters**:
  - `label` (str): Information label (e.g., 'Total steps').
  - `value`: Value to display (any printable type).
- **Returns**: None.
- **Logic**:
  - Formats the label to a fixed width and prints it with the value.
  - Used to display maze statistics and solution metrics.

### `print_path(path)`
- **Purpose**: Prints the solution path in a formatted, readable way.
- **Parameters**:
  - `path` (list): List of `(row, col)` positions in the path.
- **Returns**: None.
- **Logic**:
  - Prints the path in groups of up to 5 positions, joined by '→'.
  - Enhances readability of the solution path in console output.

### `main()`
- **Purpose**: Orchestrates the entire program, handling user interaction, search execution, and result display.
- **Parameters**: None.
- **Returns**: None.
- **Logic**:
  - Prints a program header using `print_header`.
  - Initializes the `HexGridVisualizer` and displays the original maze.
  - Generates the maze map using `generate_maze_map`.
  - Displays maze statistics (treasures, traps, rewards, obstacles).
  - Prompts for optimization mode, defaulting to 'combined' if invalid.
  - Runs A* search from `(0, 0)` using `a_star_search`.
  - If a solution is found, displays metrics, path, and visualized solution.
  - If no solution is found, explains possible reasons and redisplays the maze.
  - Entry point for the program’s execution.

## Notes
- **Performance**: The A* algorithm may take longer for complex mazes due to the need to explore multiple treasure collection orders.
- **Extensibility**: The code is modular, allowing easy addition of new trap/reward types or grid sizes by modifying the `TRAPS`, `REWARDS`, `OBSTACLES`, and `TREASURES` constants.
- **Visualization**: The hexagonal grid is offset correctly for even/odd rows, ensuring accurate neighbor calculations and visual alignment.

## Limitations
- The program assumes a fixed 10x6 grid; modifying the grid size requires updating the `HexGridVisualizer` class and maze constants.
- Trap4 causes instant path failure, which may make some maze configurations unsolvable.
- The heuristic assumes unobstructed paths for remaining treasures, which may be inadmissible in some cases but is effective for most practical mazes.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

