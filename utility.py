#!/usr/bin/env python3
"""
This is the utility file where commonly used helper
functions are stored for our Point-Scattering 
Problem implementations.
"""
# Standard libraries or third-party packages
from dataclasses import dataclass, asdict   # Used for the GA parameters
from typing import Optional
import math                                 # For calculations
import matplotlib.pyplot as plt             # For plots
from deap import base, creator, tools       # DEAP and helpers

# Local Imports
from implementations import cartesian, polar, boundary



# ===================== MATHEMATICAL FORMULAS =====================
# Insert commonly used functions here.
def in_unitCircle(x, y):
    return (x*x) + (y*y) <= 1

def calcMinEuclideanDistance(points):
    """
    Calculates the minimum Euclidean distance between all Point-Pairs
    points (p1, p2,...,pn): X,Y coordinates
    This code will refer to the following formula: d= SQRT((X_2 - X_1)^2 + (Y_2 - Y_1)^2)
    """
    min_dist = float("inf")     # Start w/ infinite distance as min

    # Iterate through all Point-Pairs
    for i in range(len(points)):
        for j in range(i + 1, len(points)): # i+1, to not include current point
            dx = points[i][0] - points[j][0]        # X-cord
            dy = points[i][1] - points[j][1]        # Y-cord

            dx_squared = dx*dx
            dy_squared = dy*dy

            d = math.sqrt((dx_squared) + (dy_squared))   # Euclidean Distance Formula

            if d < min_dist:
                min_dist = d

    return (min_dist,)

# Converts Polar coords to Cartesian
def polar_to_cart(ind):
    """
    Formula: (r*cos(theta), r*sin(theta))
    """
    conversion = []
    
    for point in ind:
        r = point[0]        # Radius
        theta = point[1]    # Angle
        
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        
        conversion.append((x, y))
    
    return conversion

# Converts Boundary coord to Cartesian
def boundary_to_cart(ind):
    """
    Formula (r=1): (cos(theta), sin(theta))
    
    Same formula as above but only 1 input variable (theta)
    """
    conversion = []
    
    for theta in ind:
        x = math.cos(theta)
        y = math.sin(theta)
        
        conversion.append((x, y))
    
    return conversion



# ===================== Plotting =====================

# Plot fitness (minimum distance) over generations
def plot_fitness_log(log, title, filename):
    generations = []
    fitness_values = []

    for entry in log:
        gen = entry[0]          # First element is generation number
        fitness = entry[1]      # Second element is fitness value
        generations.append(gen)
        fitness_values.append(fitness)
    
    # Figure setup
    plt.figure(figsize=(10, 6))
    plt.plot(generations, fitness_values, linewidth=2, color='blue')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Minimum Pairwise Distance', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add label to the last point
    best_gen = generations[-1]
    best_fitness = fitness_values[-1]
    plt.plot(best_gen, best_fitness, 'ro', markersize=8)
    plt.annotate(f'Gen {best_gen}: {best_fitness:.4f}',
                xy=(best_gen, best_fitness),
                xytext=(-80, 10),
                textcoords='offset points',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=1),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Save Plot
    plt.tight_layout()
    plt.savefig(f'graphs/{filename}', dpi=300, bbox_inches='tight')
    plt.close()

# Log point positions for each generation
def log_generation(log_file, generation, points):
    points_list = []
    for point in points:
        x = point[0]
        y = point[1]
        
        # Format the point as a string with 6 decimal places
        point_str = f'({x:.6f},{y:.6f})'
        points_list.append(point_str)
    
    # Join all strings
    points_str = ', '.join(points_list)
    
    # Write to log
    log_file.write(f"Gen {generation}: [{points_str}]\n")

# Plot points on graph with circle
def plot_point_distribution(points, title, filename):
    xs = [x[0] for x in points]
    ys = [y[1] for y in points]

    plt.figure(figsize=(6, 6))

    # Plot points
    plt.scatter(xs, ys)

    # Draw unit circle
    circle = plt.Circle((0, 0), 1.0, fill=False, linewidth=2)
    plt.gca().add_artist(circle)

    # Draw origin
    plt.plot(0, 0, 'ro', markersize=5)

    # Formatting
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('equal')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, alpha=0.3)

    # Save figure
    plt.tight_layout()
    plt.savefig(f'graphs/{filename}', dpi=300, bbox_inches='tight')
    plt.close()



# ===================== GA Configuration Dataclass =====================

# Configuration dataclass
@dataclass(frozen=True)         # Parameters cant be changed during runs
class Config:
    pop_size: int = 200         # 200 individuals
    generations: int = 200      # Allowed generations
    cxpb: float = 0.7           # crossover prob
    mutpb: float = 0.2          # mutation probability
    tournsize: int = 3          # For tournament slection
    seed: Optional[int] = None



# ===================== Running the Point-Scattering GA =====================

# class PointScatteringGA:
#     def __init__(self, n: int, representation: str="cartesian", cfg: Config=Config()):
#         self.n = n
#         self.representation = representation
#         self.cfg: Config = cfg

#         # Build the DEAP toolbox

#     def build_toolbox(self):
#         self.ensure_deap_creators()

#         ## TOOLBOX ##
#         toolbox = base.Toolbox()

#         # Operator registration
#         toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#         # cxTwoPoint choosen as it mixes parents more evenly
#         toolbox.register("mate", tools.cxTwoPoint)      
#         toolbox.register("select", tools.selTournament, tournsize=self.cfg.tournsize)   # Was the best selection method from the prev HW

#         # The 3 representations will be handled here
#         if self.representation == "cartesian":
#             # Setup the cartesian-specific operators
#             toolbox.register("individual", tools.initIterate, creator.Individual, lambda: cartesian.init_cartesian_ind(self.n))
#             toolbox.register("evaluate", cartesian.evalCartesianFitness)
#             toolbox.register("mutate", cartesian.mutate_cartesian_ind)

#         #elif self.representation == "polar":

#         #elif self.representation == "angle":

#         else:
#             raise ValueError(f"Unknown representation: {self.representation}")

#         self.toolbox = toolbox
    
#     # Method which ensure that no duplicates are existing
#     def ensure_deap_creators(self) -> None:
#         if not hasattr(creator, "FitnessMax"):      # Ensure doesnt exist for both FitnessMax & Individual
#             creator.create("FitnessMax", base.Fitness, weights=(1.0,))
#         if not hasattr(creator, "Individual"):
#             creator.create("Individual", list, fitness=creator.FitnessMax)
    
#     def runGA(self):
#         return
