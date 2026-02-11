#!/usr/bin/env python3
"""
This is the utility file where commonly used helper
functions are stored for our Point-Scattering 
Problem implementations.
"""
# Standard libraries or third-party packages
import math                                 # For calculations
import numpy as np
import matplotlib.pyplot as plt             # For plots
from scipy import stats       
from dataclasses import dataclass, asdict   # Used for the GA parameters
from typing import List, Optional
from deap import base, creator, tools       # DEAP and helpers


# Local Imports
from implementations import cartesian, polar, boundary



# ===================== MATHEMATICAL FORMULAS =====================
# Insert commonly used functions here.
def in_unitCircle(x, y):
    return (x*x) + (y*y) <= 1

# Calculates using Cartesian Points
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

            # Euclidean Distance Formula
            d = math.sqrt((dx_squared) + (dy_squared))

            if d < min_dist:
                min_dist = d

    return (min_dist,)

# Calculates using Polar Points
def calcMinEuclideanDistancePolar(points):
    """
    Calculates the minimum Euclidean distance between all Point-Pairs
    points (p1, p2,...,pn): r,theta coordinates
    This code will refer to the following formula: d= SQRT(r1^2 + r2^2 - 2*r1*r2*cos(theta2 - theta1)
    """
    min_dist = float("inf")     # Start w/ infinite distance as min

    # Iterate through all Point-Pairs
    for i in range(len(points)):
        for j in range(i + 1, len(points)): # i+1, to not include current point
            r1 = points[i][0]
            theta1 = points[i][1]
            r2 = points[j][0]
            theta2 = points[j][1]

            r1_squared = r1 * r1
            r2_squared = r2 * r2
            
            # Euclidean Distance Formula
            d = math.sqrt(r1_squared + r2_squared - (2*r1*r2*math.cos(theta1 - theta2)))

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



# ===================== STATS =====================
def mean_std_ci95(values: List[float]) -> tuple[float, float, tuple[float, float]]:
    """
    Statistical information for final fitness
    """
    arr = np.array(values)
    n = len(arr)                # Sample size
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    
    # Calculate 95% CI using the t-distribution
    # df = degrees of freedom (n-1)
    # loc = sample mean
    # scale = standard error of the sample mean
    sem = stats.sem(arr)
    ci_low, ci_high = stats.t.interval(confidence=0.95, df=n-1, loc=mean, scale=sem)
    
    return mean, std, (ci_low, ci_high)

def per_gen_mean_ci(series_2d: List[List[float]]):      
    arr = np.array(series_2d, dtype=float)
    n = arr.shape[0]            
    mean = np.mean(arr, axis=0)

    # Calculate standard error along axis 0
    sem = stats.sem(arr, axis=0)

    # Calculate 95% CI for the whole array at once
    # This returns two arrays: (Lower Bounds, Upper Bounds)
    ci_low, ci_high = stats.t.interval(confidence=0.95, df=n-1, loc=mean, scale=sem)
    
    return mean, ci_low, ci_high

def print_results(representation: str, results: dict):
    print(f"Printing {representation} results........")
    # Prints the final stat results
    stats = results["final_stats"]
    mean_f = stats["mean"]
    std_f = stats["std"]
    CI_low, CI_high = stats["CI95"]
    best_overall_fitness = max(results["best_overall_all"])

    print(f"\n{representation} final stats:")
    print(f"MEAN: {mean_f:.3f}")
    print(f"STD: {std_f:.3f}")
    print(f"95% Confidence Interval: {CI_low:.3f}, {CI_high:.3f}")
    print(f"\tFinal Best Minimum Distance: {best_overall_fitness:.6f}\n")

    gen_stats = results["gen_stats"]
    gen_m = gen_stats["mean"]

    print(f"Best mean fitness at final generation: {gen_m[-1]:.3f}")    # [-1] gets final item



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
    cxpb: float = 0.8           # crossover prob
    mutpb: float = 0.2          # mutation probability
    tournsize: int = 3          # For tournament slection
    seed: Optional[int] = None