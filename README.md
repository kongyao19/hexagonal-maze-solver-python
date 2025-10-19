# Hexagonal Maze A* Solver 🗺️

A Python-based hexagonal maze solver that demonstrates artificial intelligence pathfinding using the A* search algorithm with dynamic cost systems, environmental hazards, and interactive visualization.

## Features

- **Hexagonal Grid Navigation**: Six-directional movement with proper offset coordinate handling for even/odd rows
- **A* Search Algorithm**: Informed search with admissible heuristic for guaranteed optimal solutions
- **Dynamic Cost System**: Movement costs modified by environmental effects (traps and rewards)
- **Multiple Optimization Modes**: Minimize steps, energy consumption, or balanced combination
- **Complex Trap Mechanics**: Energy multipliers, step modifiers, teleportation, and path invalidation
- **Reward System**: Beneficial effects that reduce movement costs
- **Interactive Visualization**: Beautiful matplotlib-based hexagonal grid with animated solution paths
- **Permutation-Based Heuristic**: Evaluates all treasure collection orderings for optimal cost estimation
- **State Management**: Comprehensive state tracking with effect stacking and energy constraints

## Tech Stack

- **Language**: Python 3.7+
- **Libraries**: matplotlib, heapq, itertools
- **Core Concepts Applied**:
  - A* search algorithm with priority queues
  - Admissible and consistent heuristic functions
  - Hexagonal grid mathematics (cube coordinate conversion)
  - State space search and graph traversal
  - Dynamic programming for path optimization
  - Object-oriented design with state encapsulation

## How to Run

**Prerequisites**: Ensure you have Python 3.7+ and pip installed

```bash
# Check Python version
python --version
```

**Clone the repository**:

```bash
git clone https://github.com/kongyao19/hexagonal-maze-solver-python.git
cd hexagonal-maze-solver-python
```

**Install dependencies**:

```bash
pip install matplotlib
```

**Run the solver**:

```bash
python hexagonal_maze_solver.py
```

**Select optimization mode** when prompted:
- Type `steps` to minimize total steps taken
- Type `energy` to minimize energy consumption  
- Type `combined` for balanced optimization (default)

The program will display the original maze, execute the search, and visualize the optimal solution path.

## Game Rules

### Maze Components

| Component | Symbol | Effect |
|-----------|--------|--------|
| **Trap 1** | ⊖ | Doubles energy cost for future moves |
| **Trap 2** | ⊕ | Doubles step count for future moves |
| **Trap 3** | ⊗ | Teleports player 2 steps forward in last direction |
| **Trap 4** | ⊘ | Invalidates entire path (instant failure) |
| **Reward 1** | ⊞ | Halves energy cost for future moves |
| **Reward 2** | ⊠ | Halves step count for future moves |
| **Treasure** | 🟠 | Must collect all 4 to complete maze |
| **Obstacle** | ⬛ | Blocks movement completely |

### Gameplay

- Player starts at entry point (0, 0) with 50 energy units
- Goal: Collect all 4 treasures optimally while avoiding Trap 4
- Each move costs 1 step and 1 energy unit (base cost)
- Effects from traps/rewards modify movement costs dynamically
- Game ends when all treasures collected or path becomes invalid

### Algorithm Details

**A* Evaluation Function**:
```
f(n) = g(n) + h(n)
```
- **g(n)**: Actual cost from start (steps, energy, or combined)
- **h(n)**: Heuristic estimate using hexagonal distance and permutation optimization

**Hexagonal Distance Calculation**:
- Converts offset coordinates to cube coordinates
- Uses Manhattan distance in 3D cube space: `max(|x₁-x₂|, |y₁-y₂|, |z₁-z₂|)`

**Heuristic Properties**:
- Admissible: Never overestimates true cost
- Consistent: Satisfies triangle inequality
- Optimal: Evaluates all treasure collection orderings

### Winning Strategy

The algorithm finds optimal paths by:
1. Prioritizing states with lowest f(n) scores
2. Considering all possible treasure collection orders
3. Applying current effects to cost estimates
4. Avoiding Trap 4 and managing energy constraints
5. Balancing steps and energy based on selected mode

## Example Output

```
[SUCCESS] Found solution after 847 iterations

Solution Found!
Optimal path successfully computed with the following metrics:
  - Total steps taken   : 18.5
  - Total energy used   : 17.0
  - Remaining energy    : 33.0
  - Path length         : 21
  - Treasures collected : 4
  - Treasure positions  : [(3, 4), (4, 1), (7, 3), (9, 3)]
  - Active effects      : ['half_energy', 'half_step']

Path:
  (0, 0) → (1, 0) → (2, 0) → (3, 0) → (4, 0)
  (4, 1) → (5, 1) → (5, 2) → (6, 2) → (7, 2)
  ...
```

## Learning Outcomes

This project demonstrates practical application of artificial intelligence and search algorithms:

- **Search Algorithms**: Implementing A* with proper evaluation functions and heuristics
- **Graph Theory**: Hexagonal grid representation and traversal with offset coordinates
- **State Space Search**: Managing complex state spaces with multiple constraints
- **Heuristic Design**: Creating admissible heuristics for optimal pathfinding
- **Algorithm Optimization**: Using priority queues and visited state tracking for efficiency
- **Data Structures**: Leveraging heaps, sets, and custom classes for algorithm implementation
- **Visualization**: Creating intuitive graphical representations of search results

The challenges faced during development, particularly with hexagonal grid mathematics, effect stacking, and heuristic optimization, provided valuable experience in AI algorithm design and implementation.

## Credits

**Developer**:
- Kong Yao 
- Lee Shu Ann 
- Lim Yi 
- Samuel Lee Zhi Jian 
- Swee Shi Yi 
- Tong Zi Qian
 
**Institution**: Sunway University, Faculty of Engineering and Technology, BSc (Hons) Computer Science  
**Course**: CSC3206 - Artificial Intelligence  
**Academic Session**: April 2025  

---

*This project was developed as part of the Artificial Intelligence course, showcasing the practical application of search algorithms in complex pathfinding scenarios.*
