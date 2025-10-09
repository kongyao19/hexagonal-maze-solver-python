#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 16:57:14 2025

@author: samuellee
"""

# Hexagonal Maze Treasure Hunt Solver using A* Algorithm with Visualization
"""
This program solves a hexagonal maze treasure hunt problem using the A* search algorithm.
The maze contains treasures to collect, traps and rewards that affect movement costs,
and obstacles that block movement. The goal is to find an optimal path that collects
all treasures while minimizing steps, energy, or a combination of both.

Key Features:
- Hexagonal grid navigation with proper neighbor calculation
- Multiple trap types with different effects (energy cost, step cost, teleportation, path invalidation)
- Reward types that reduce costs
- A* pathfinding with multiple optimization modes
- Interactive visualization of the maze and solution path
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import heapq
from itertools import permutations

# ==================== MAZE CONFIGURATION AND CONSTANTS ====================
"""
Define all the special locations in the hexagonal maze.
Coordinates are in (row, col) format where (0,0) is the top-left corner.
"""

# Treasure locations - these must all be collected to complete the maze
TREASURES = {(3, 4), (4, 1), (7, 3), (9, 3)}

# Trap definitions with their specific effects
TRAPS = {
    'Trap1': {(8, 2)},        # Double energy cost for future moves
    'Trap2': {(1, 1), (2, 4)}, # Double step cost for future moves  
    'Trap3': {(5, 3), (6, 1)}, # Teleport two steps in last move direction
    'Trap4': {(3, 1)},        # Invalidates the entire path (instant failure)
}

# Reward definitions that provide beneficial effects
REWARDS = {
    'Reward1': {(1, 3), (4, 0)}, # Half energy cost for future moves
    'Reward2': {(5, 5), (7, 2)}, # Half step cost for future moves
}

# Obstacle locations that cannot be traversed
OBSTACLES = {(0, 3), (2, 2), (3, 3), (4, 2), (4, 4), (6, 3), (6, 4), (7, 4), (8, 1)}

# Direction mappings for hexagonal grids (different for even/odd rows)
HEX_DIRECTIONS_EVEN = {
    "up": (0, -1), "down": (0, 1),
    "upper-left": (-1, 0), "upper-right": (1, 0),
    "bottom-left": (-1, 1), "bottom-right": (1, 1),
}

HEX_DIRECTIONS_ODD = {
    "up": (0, -1), "down": (0, 1),
    "upper-left": (-1, -1), "upper-right": (1, -1),
    "bottom-left": (-1, 0), "bottom-right": (1, 0),
}

# ==================== STATE CLASS DEFINITION ====================
class State:
    """
    Represents a state in the search space for the A* algorithm.
    Each state contains all information needed to evaluate and continue the search.
    """
    
    def __init__(self, position, path=None, steps=0, energy=0, treasures_collected=None, effects=None, direction=None, remaining_energy=50):
        """
        Initialize a new state.
        
        Args:
            position (tuple): Current (row, col) position in the maze
            path (list): List of positions visited to reach this state
            steps (int): Total number of steps taken to reach this state
            energy (int): Total energy consumed to reach this state
            treasures_collected (set): Set of treasure positions already collected
            effects (set): Set of active effects from traps/rewards
            direction (str): Last movement direction taken to reach this state
            remaining_energy (int): Initially set as 50
        """
        self.position = position
        self.path = path if path is not None else [position]
        self.steps = steps
        self.energy = energy
        self.treasures_collected = treasures_collected if treasures_collected is not None else set()
        self.effects = effects if effects is not None else set()
        self.direction = direction
        self.remaining_energy = remaining_energy

    def __lt__(self, other):
        """
        Comparison function for priority queue ordering in A*.
        States with lower steps and energy are considered "less than" others.
        """
        return (self.steps, self.energy) < (other.steps, other.energy)

# ==================== VISUALIZATION CLASS ====================
class HexGridVisualizer:
    """
    Handles visualization of the hexagonal maze using matplotlib.
    Creates an interactive display showing the maze layout, special cells,
    and optionally the solution path.
    """
    
    def __init__(self):
        """Initialize the visualizer with grid dimensions and visual settings."""
        # Grid dimensions (note: these are swapped from the maze coordinates)
        self.rows = 6    # Number of rows in the visual grid
        self.cols = 10   # Number of columns in the visual grid
        
        # Cell type constants for internal representation
        self.EMPTY, self.TRAP1, self.TRAP2, self.TRAP3, self.TRAP4 = 0, 1, 2, 3, 4
        self.REWARD1, self.REWARD2, self.TREASURE, self.OBSTACLE = 5, 6, 7, 8
        
        # Color mapping for different cell types
        self.colors = {
            self.EMPTY: 'white',        # Empty cells
            self.TRAP1: '#DDA0DD',      # All traps use purple variants
            self.TRAP2: '#DDA0DD', 
            self.TRAP3: '#DDA0DD', 
            self.TRAP4: '#DDA0DD',
            self.REWARD1: '#40E0D0',    # All rewards use turquoise variants
            self.REWARD2: '#40E0D0',
            self.TREASURE: '#FFA500',   # Treasures are orange
            self.OBSTACLE: '#696969'    # Obstacles are gray
        }
        
        # Symbol mapping for special cells (displayed as text overlays)
        self.symbols = {
            self.TRAP1: '⊖',     # Trap 1 symbol
            self.TRAP2: '⊕',     # Trap 2 symbol  
            self.TRAP3: '⊗',     # Trap 3 symbol
            self.TRAP4: '⊘',     # Trap 4 symbol
            self.REWARD1: '⊞',   # Reward 1 symbol
            self.REWARD2: '⊠',   # Reward 2 symbol
        }
        
        # Initialize the grid with all cell types
        self.grid = self._initialize_grid()

    def _initialize_grid(self):
        """
        Initialize the visualization grid by mapping maze coordinates to cell types.
        
        Returns:
            dict: Mapping from (col, row) coordinates to cell type constants
        """
        grid = {}
        
        # Start with all cells as empty
        for col in range(self.cols):
            for row in range(self.rows):
                grid[(col, row)] = self.EMPTY
        
        # Populate special cell types based on maze metadata
        # Note: The maze uses (row, col) but visualization uses (col, row)
        for pos in TREASURES:
            grid[pos] = self.TREASURE
            
        # Add all trap types
        for pos in TRAPS['Trap1']:
            grid[pos] = self.TRAP1
        for pos in TRAPS['Trap2']:
            grid[pos] = self.TRAP2
        for pos in TRAPS['Trap3']:
            grid[pos] = self.TRAP3
        for pos in TRAPS['Trap4']:
            grid[pos] = self.TRAP4
            
        # Add all reward types
        for pos in REWARDS['Reward1']:
            grid[pos] = self.REWARD1
        for pos in REWARDS['Reward2']:
            grid[pos] = self.REWARD2
            
        # Add obstacles
        for pos in OBSTACLES:
            grid[pos] = self.OBSTACLE
            
        return grid

    def get_hex_position(self, col, row):
        """
        Calculate the (x, y) pixel position for a hexagon in the visualization.
        Uses proper hexagonal grid spacing with offset rows.
        
        Args:
            col (int): Column index in the grid
            row (int): Row index in the grid
            
        Returns:
            tuple: (x, y) position in the matplotlib coordinate system
        """
        # Hexagonal grid positioning math
        x = col * 1.4 * math.sqrt(3) * 0.5  # Horizontal spacing
        y = row * math.sqrt(3) * 1.05        # Vertical spacing
        
        # Offset every other column for proper hexagonal alignment
        if col % 2 == 0:
            y += math.sqrt(3) * 0.525
            
        return x, -y  # Negative y to flip coordinate system

    def visualize(self, path=None, title=None):
        """
        Create and display the visualization of the hexagonal maze.
        
        Args:
            path (list, optional): List of (col, row) positions showing the solution path
            title (str, optional): Custom title for the visualization
        """
        # Create the matplotlib figure and axis
        fig, ax = plt.subplots(figsize=(18, 12))
        hex_radius = 0.68  # Size of each hexagon
        
        # Draw all hexagons in the grid
        for (col, row), cell_type in self.grid.items():
            x, y = self.get_hex_position(col, row)
            
            # Create hexagon patch with appropriate color
            hexagon = patches.RegularPolygon(
                (x, y), numVertices=6, radius=hex_radius, orientation=math.radians(30),
                facecolor=self.colors[cell_type], edgecolor='black', linewidth=1.5
            )
            ax.add_patch(hexagon)
            
            # Add coordinate labels to each hexagon
            ax.text(x, y + 0.4, f"({col},{row})", ha='center', va='center', 
                   fontsize=9, weight='bold')
            
            # Add special symbols for traps and rewards
            if cell_type in self.symbols:
                ax.text(x, y, self.symbols[cell_type], ha='center', va='center', 
                       fontsize=18, weight='bold')

        # Draw the solution path if provided
        if path:
            # Draw arrows between consecutive path positions
            for i in range(len(path) - 1):
                x1, y1 = self.get_hex_position(*path[i])
                x2, y2 = self.get_hex_position(*path[i + 1])
                ax.arrow(x1, y1, x2-x1, y2-y1, head_width=0.15, head_length=0.15, 
                        fc='red', ec='red', alpha=0.7, linewidth=3, zorder=2)
            
            # Mark start position with green circle and 'S'
            start_x, start_y = self.get_hex_position(*path[0])
            ax.add_patch(patches.Circle((start_x, start_y), radius=0.3, color='green', zorder=1))
            ax.text(start_x, start_y, 'S', ha='center', va='center', 
                   fontsize=14, color='white', weight='bold')
            
            # Mark end position with blue circle and 'E'
            end_x, end_y = self.get_hex_position(*path[-1])
            ax.add_patch(patches.Circle((end_x, end_y), radius=0.3, color='blue', zorder=1))
            ax.text(end_x, end_y, 'E', ha='center', va='center', 
                   fontsize=14, color='white', weight='bold')

        # Add entry point arrow at (0,0)
        start_x, start_y = self.get_hex_position(0, 0)
        ax.arrow(start_x - 1.4, start_y, 0.8, 0, head_width=0.3, head_length=0.3, 
                fc='blue', ec='blue')
        ax.text(start_x - 1.7, start_y, 'Entry', color='blue', weight='bold', fontsize=12)

        # Create legend showing all cell types
        legend_x = self.cols * math.sqrt(3) * 0.7 + 1.5  # Position legend to the right
        legend_y = 0.5
        dy = 0.9  # Vertical spacing between legend items
        legend_hex_radius = 0.35

        ax.text(legend_x, legend_y + dy, "Legend", fontsize=14, weight='bold', ha='left')

        # Define legend items with their labels
        legend_items = [
            (self.TRAP1, 'Trap 1'), (self.TRAP2, 'Trap 2'), 
            (self.TRAP3, 'Trap 3'), (self.TRAP4, 'Trap 4'),
            (self.REWARD1, 'Reward 1'), (self.REWARD2, 'Reward 2'), 
            (self.TREASURE, 'Treasure'), (self.OBSTACLE, 'Obstacle')
        ]

        # Draw each legend item
        for i, (cell_type, label) in enumerate(legend_items):
            y_pos = legend_y - i * dy
            
            # Create small hexagon for legend
            hex_patch = patches.RegularPolygon(
                (legend_x, y_pos), numVertices=6, radius=legend_hex_radius, 
                orientation=math.radians(30), facecolor=self.colors[cell_type], 
                edgecolor='black', linewidth=1
            )
            ax.add_patch(hex_patch)

            # Add symbol if applicable
            if cell_type in self.symbols:
                ax.text(legend_x, y_pos, self.symbols[cell_type], ha='center', 
                       va='center', fontsize=12, weight='bold')

            # Add label text
            ax.text(legend_x + 0.8, y_pos, label, va='center', fontsize=12)

        # Set up the plot area and display
        ax.set_xlim(-2.5, legend_x + 2.5)
        ax.set_ylim(-self.rows * math.sqrt(3) * 1.2, legend_y + dy * 2)
        ax.set_aspect('equal')  # Maintain aspect ratio for proper hexagon shape
        ax.axis('off')          # Hide axis lines and ticks
        
        # Set title
        if title:
            ax.set_title(title, fontsize=30, weight='bold')
        else:
            default_title = "Map with Optimal Path" if path else "Map of the Virtual World"
            ax.set_title(default_title, fontsize=30, weight='bold')
        
        plt.tight_layout()
        plt.show()

# ==================== MAZE UTILITY FUNCTIONS ====================
def get_neighbors(rows, cols, pos, obstacles):
    """
    Calculate valid neighboring positions for a given position in the hexagonal grid.
    Hexagonal grids have 6 neighbors, with positions depending on whether
    the current row is even or odd due to the offset pattern.

    Args:
        rows (int): Total number of rows in the maze
        cols (int): Total number of columns in the maze  
        pos (tuple): Current (row, col) position
        obstacles (set): Set of obstacle positions to avoid

    Returns:
        list: List of valid neighboring (row, col) positions
    """
    r, c = pos

    # Use even or odd direction mapping
    directions = HEX_DIRECTIONS_EVEN if r % 2 == 0 else HEX_DIRECTIONS_ODD

    neighbors = []
    for dr, dc in directions.values():
        nr, nc = r + dr, c + dc
        
        # Filter out invalid positions (out of bounds or obstacles)
        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in obstacles:
            neighbors.append((nr, nc))
    
    return neighbors

def generate_maze_map():
    """
    Generate a complete maze map with all cell information and valid moves.
    This creates the data structure used by the pathfinding algorithm.
    
    Returns:
        dict: Mapping from (row, col) to cell information including content and valid moves
    """
    maze_map = {}
    
    # Process each cell in the 10x6 maze grid
    for row in range(10):
        for col in range(6):
            pos = (row, col)
            
            # Determine cell content based on maze metadata
            content = 'Empty'  # Default content
            
            if pos in TREASURES:
                content = 'Treasure'
            elif pos in TRAPS['Trap1']:
                content = 'Trap1'
            elif pos in TRAPS['Trap2']:
                content = 'Trap2'
            elif pos in TRAPS['Trap3']:
                content = 'Trap3'
            elif pos in TRAPS['Trap4']:
                content = 'Trap4'
            elif pos in REWARDS['Reward1']:
                content = 'Reward1'
            elif pos in REWARDS['Reward2']:
                content = 'Reward2'
            elif pos in OBSTACLES:
                content = 'Obstacle'
            
            # Store cell information
            maze_map[pos] = {
                'content': content,
                'moves': get_neighbors(10, 6, pos, OBSTACLES)  # Valid moves from this position
            }
    
    return maze_map

# ==================== HEXAGONAL GRID DIRECTION HANDLING ====================
def apply_direction(pos, direction_name):
    """
    Apply a named direction to a position to get the next position.
    Used for Trap3 teleportation logic.
    
    Args:
        pos (tuple): Starting (row, col) position
        direction_name (str): Direction name (e.g., "up", "bottom-left")
        
    Returns:
        tuple: New (row, col) position after applying the direction
    """
    r, c = pos
    # Choose direction mapping based on row parity
    dirs = HEX_DIRECTIONS_EVEN if r % 2 == 0 else HEX_DIRECTIONS_ODD
    dr, dc = dirs[direction_name]
    return (r + dr, c + dc)

def find_direction_name(r0, c0, r1, c1):
    """
    Determine the direction name for moving from one position to another.
    Used to identify movement direction for Trap3 teleportation.
    
    Args:
        r0, c0 (int): Starting row and column
        r1, c1 (int): Ending row and column
        
    Returns:
        str or None: Direction name if valid move, None otherwise
    """
    # Choose direction mapping based on starting row parity
    dirs = HEX_DIRECTIONS_EVEN if r0 % 2 == 0 else HEX_DIRECTIONS_ODD
    
    # Check each direction to find a match
    for name, (dr, dc) in dirs.items():
        if (r0 + dr, c0 + dc) == (r1, c1):
            return name
    return None

# ==================== A* ALGORITHM IMPLEMENTATION ====================
def get_successors(state, maze_map):
    """
    Generate all valid successor states from the current state.
    This is the core function that handles movement, trap effects, and state transitions.
    
    Args:
        state (State): Current state in the search
        maze_map (dict): Complete maze information
        
    Returns:
        list: List of valid successor State objects
    """
    successors = []
    
    # Try moving to each valid neighbor
    for neighbor in maze_map[state.position]["moves"]:
        # Determine the direction of movement
        dir_name = find_direction_name(*state.position, *neighbor)
        if not dir_name:
            continue  # Skip if direction cannot be determined

        # Get the content of the destination cell
        content = maze_map.get(neighbor, {}).get("content", "Empty")

        # Calculate movement costs based on current effects
        step_cost = 1 * step_multiplier(state.effects)
        energy_cost = 1 * energy_multiplier(state.effects)

        # Initialize new state values
        new_steps = state.steps + step_cost
        new_energy = state.energy + energy_cost
        new_effects = state.effects.copy()
        new_treasures = state.treasures_collected.copy()
        new_path = state.path + [neighbor]
        new_pos = neighbor

        # Handle Trap3 special case: teleportation
        if content == 'Trap3':
            # Trap3 teleports the player two steps in the same direction
            step1 = apply_direction(neighbor, dir_name)      # First teleport step
            step2 = apply_direction(step1, dir_name)         # Second teleport step
            
            # Check if both teleportation steps are valid
            if (step1 not in maze_map or maze_map[step1]['content'] == 'Obstacle' or
                step2 not in maze_map or maze_map[step2]['content'] == 'Obstacle'):
                continue  # Skip this move if teleportation leads to invalid position
               
            # Update position and path for teleportation
            new_pos = step2
            content = maze_map[new_pos]['content']  # Update content to final destination
            new_path = state.path + [neighbor, step2]  # Include intermediate position

        # Apply effects based on final destination content
        if content == 'Trap1':
            new_effects.add('double_energy')  # Future moves cost double energy
        elif content == 'Trap2':
            new_effects.add('double_step')    # Future moves cost double steps
        elif content == 'Trap4':
            return []  # Trap4 invalidates the entire path - no successors
        elif content == 'Reward1':
            new_effects.add('half_energy')    # Future moves cost half energy
        elif content == 'Reward2':
            new_effects.add('half_step')      # Future moves cost half steps
        elif content == 'Treasure':
            new_treasures.add(new_pos)        # Collect the treasure
        # Calculate remaining energy
        new_remaining_energy = state.remaining_energy - energy_cost
        
        # Skip this move if we don't have enough energy
        if new_remaining_energy < 0:
            continue
        
        # Create and add the successor state
        successors.append(State(
            position=new_pos,
            path=new_path,
            steps=new_steps,
            energy=new_energy,
            treasures_collected=new_treasures,
            effects=new_effects,
            direction=dir_name,
            remaining_energy=new_remaining_energy
        ))
    
    return successors

# ==================== HEURISTIC FUNCTIONS ====================
def hex_distance(a, b):
    """
    Calculate the shortest distance between two points in a hexagonal grid.
    Uses cube coordinate conversion for accurate hexagonal distance calculation.
    
    Args:
        a, b (tuple): Two (row, col) positions
        
    Returns:
        int: Minimum number of steps between the positions
    """
    def to_cube(r, c):
        """Convert offset coordinates to cube coordinates."""
        x = c - (r + (r & 1)) // 2  # Convert to cube x
        z = r                        # Cube z is same as row
        y = -x - z                   # Cube coordinates sum to zero
        return (x, y, z)
    
    # Convert both positions to cube coordinates
    ax, ay, az = to_cube(*a)
    bx, by, bz = to_cube(*b)
    
    # Hexagonal distance is the maximum of the three coordinate differences
    return max(abs(ax - bx), abs(ay - by), abs(az - bz))

def step_multiplier(effects):
    """
    Calculate the step cost multiplier based on active effects.
    
    Args:
        effects (set): Set of active effect names
        
    Returns:
        float: Multiplier for step costs (0.5, 1.0, 2.0, etc.)
    """
    mult = 1.0
    if 'double_step' in effects:
        mult *= 2
    if 'half_step' in effects:
        mult *= 0.5
    return mult

def energy_multiplier(effects):
    """
    Calculate the energy cost multiplier based on active effects.
    
    Args:
        effects (set): Set of active effect names
        
    Returns:
        float: Multiplier for energy costs (0.5, 1.0, 2.0, etc.)
    """
    mult = 1.0
    if 'double_energy' in effects:
        mult *= 2
    if 'half_energy' in effects:
        mult *= 0.5
    return mult

def heuristic(state, maze_map, mode='steps'):
    """
    Calculate the heuristic value for A* search.
    Estimates the minimum cost to collect all remaining treasures.
    Uses permutation-based optimization to find the best collection order.
    
    Args:
        state (State): Current state
        maze_map (dict): Complete maze information
        mode (str): Optimization mode ('steps', 'energy', or 'combined')
        
    Returns:
        float: Estimated minimum cost to complete the goal
    """
    def estimate_path_cost(start, remaining, effects):
        """
        Estimate the minimum cost to visit all remaining treasures.
        Tries different visit orders and returns the best estimate.
        """
        if not remaining:
            return 0
            
        best = float('inf')
        
        # Try all possible orders of visiting remaining treasures
        for order in permutations(remaining):
            pos = start
            eff = effects.copy()
            s_total = 0  # Total steps
            e_total = 0  # Total energy
            
            # Calculate cost for this particular order
            for treasure_pos in order:
                # Estimate distance and costs to this treasure
                dist = hex_distance(pos, treasure_pos)
                s_total += dist * step_multiplier(eff)
                e_total += dist * energy_multiplier(eff)
                pos = treasure_pos
            
            # Calculate total cost based on optimization mode
            if mode == 'steps':
                cost = s_total
            elif mode == 'energy':
                cost = e_total
            elif mode == 'combined':
                cost = s_total + e_total
            
            best = min(best, cost)
        
        return best

    # Find all uncollected treasures
    remaining = [pos for pos, cell in maze_map.items() 
                if cell['content'] == 'Treasure' and pos not in state.treasures_collected]
    
    # Return estimated cost to collect all remaining treasures
    return estimate_path_cost(state.position, remaining, state.effects.copy())

def a_star_search(start, maze_map, total_treasures, mode='combined', verbose=False):
    """
    Perform A* search to find the optimal path that collects all treasures.
    
    Args:
        start (State): Starting state
        maze_map (dict): Complete maze information
        total_treasures (int): Total number of treasures to collect
        mode (str): Optimization mode ('steps', 'energy', or 'combined')
        verbose (bool): Whether to print search progress
        
    Returns:
        State or None: Goal state if solution found, None otherwise
    """
    # Priority queue for A* search: (f_cost, state)
    frontier = [(0, start)]
    
    # Visited states tracker: maps state key to best known cost
    visited = {}
    
    # Search statistics
    iterations = 0
    
    while frontier:
        iterations += 1
        
        # Get the state with lowest f-cost (g + h)
        _, current = heapq.heappop(frontier)

        # Check if goal reached (all treasures collected)
        if len(current.treasures_collected) == total_treasures:
            if verbose:
                print("\n[SUCCESS] Found solution after {} iterations".format(iterations))
            return current
        
        # Create state key for duplicate detection
        # Key includes position, treasures collected, and active effects
        key = (current.position, frozenset(current.treasures_collected), frozenset(current.effects), current.remaining_energy)
        
        # Calculate g-cost based on optimization mode
        if mode == 'steps':
            score = current.steps
        elif mode == 'energy':
            score = current.energy
        else:  # combined
            score = current.steps + current.energy
        
        # Skip if we've seen this state with better or equal cost
        if key in visited and visited[key] <= score:
            continue
        visited[key] = score

        # Generate and process all successor states
        for succ in get_successors(current, maze_map):
            # Calculate heuristic for this successor
            h = heuristic(succ, maze_map, mode)
            
            # Calculate g-cost for this successor
            if mode == 'steps':
                g = succ.steps
            elif mode == 'energy':
                g = succ.energy
            else:  # combined
                g = succ.steps + succ.energy
            
            # Add to frontier with f-cost = g + h
            heapq.heappush(frontier, (g + h, succ))
    
    # No solution found
    if verbose:
        print("\n[FAILURE] Search completed after {} iterations, no solution found".format(iterations))
    return None

# ==================== MAIN PROGRAM ====================

def print_header(text):
    """Print a formatted header for program sections."""
    print("\n" + "=" * 80)
    print(" " * ((80 - len(text)) // 2) + text)
    print("=" * 80)

def print_section(text):
    """Print a formatted section header."""
    print("\n[+] {}".format(text))

def print_info(label, value):
    """Print formatted information with label and value."""
    print("  - {:20}: {}".format(label, value))

def print_path(path):
    """Print the solution path in a formatted way."""
    print("\nPath:")
    # Print path positions in groups of 5 for readability
    for i in range(0, len(path), 5):
        print("  " + " → ".join(str(pos) for pos in path[i:i+5]))

def main():
    """
    Main function that orchestrates the entire maze solving process.
    Handles user interaction, executes the search algorithm, and displays results.
    """
    # Print program header with title
    print_header("HEXAGONAL MAZE TREASURE HUNT SOLVER")
    print("Using A* Algorithm with Visualization\n")
    
    # Initialize visualizer and show the original maze layout
    visualizer = HexGridVisualizer()
    print_section("Displaying original map...")
    visualizer.visualize(title="Original Hexagonal Maze")
    
    # Generate the maze data structure for pathfinding algorithm
    maze_map = generate_maze_map()
    total_treasures = sum(1 for cell in maze_map.values() if cell['content'] == 'Treasure')
    
    # Display maze statistics and configuration
    print_section("Maze Information")
    print_info("Total treasures", total_treasures)
    print_info("Total traps", sum(len(t) for t in TRAPS.values()))
    print_info("Total rewards", sum(len(r) for r in REWARDS.values()))
    print_info("Total obstacles", len(OBSTACLES))
    
    # Get optimization mode from user input
    print_section("Search Configuration")
    print("Available optimization modes:")
    print("  - 'steps': Minimize total number of steps taken")
    print("  - 'energy': Minimize total energy consumed") 
    print("  - 'combined': Minimize steps + energy (balanced approach)")
    
    mode = input("\nEnter optimization mode (steps/energy/combined): ").strip().lower()
    if mode not in ['steps', 'energy', 'combined']:
        print("Invalid mode entered, defaulting to 'combined'")
        mode = 'combined'
    
    # Initialize starting state at position (0, 0)
    start_state = State(position=(0, 0))
    
    # Execute A* search algorithm
    print_section("Starting A* search from (0, 0) with mode '{}'".format(mode))
    print("Searching for optimal path to collect all treasures...")
    print("This may take a moment depending on maze complexity...")
    
    # Run the search
    goal_state = a_star_search(start_state, maze_map, total_treasures, mode=mode, verbose=True)

    # Process and display results
    if goal_state:
        # Solution found - display detailed results
        print_section("Solution Found!")
        print("Optimal path successfully computed with the following metrics:")
        print_info("Total steps taken", goal_state.steps)
        print_info("Total energy used", goal_state.energy)
        print_info("Remaining energy", goal_state.remaining_energy)
        print_info("Path length", len(goal_state.path))
        print_info("Treasures collected", len(goal_state.treasures_collected))
        print_info("Treasure positions", sorted(goal_state.treasures_collected))
        
        # Display active effects at the end of the path
        if goal_state.effects:
            print_info("Active effects", sorted(goal_state.effects))
        else:
            print_info("Active effects", "None")
        
        # Print the complete path taken
        print_path(goal_state.path)
        
        # Show visualization with the solution path highlighted
        print_section("Displaying solution path...")
        print("Green circle (S) = Start position")
        print("Blue circle (E) = End position") 
        print("Red arrows = Movement path")
        visualizer.visualize(path=goal_state.path, 
                           title="Optimal Path (Mode: {})".format(mode.capitalize()))
        
    else:
        # No solution found
        print_section("No Solution Found")
        print("The search algorithm was unable to find a valid path that collects all treasures.")
        print("This could be due to:")
        print("  - Treasures being unreachable due to obstacles")
        print("  - Trap4 locations blocking all possible paths")
        print("  - Maze configuration making treasure collection impossible")
        
        # Still display the original map for reference
        print_section("Displaying original map for reference...")
        visualizer.visualize(title="No Solution Found - Original Maze")

# ==================== PROGRAM ENTRY POINT ====================
if __name__ == "__main__":
    """
    Program entry point. Only runs main() if script is executed directly,
    not when imported as a module.
    """
    main()