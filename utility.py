#!/usr/bin/env python3
"""
This is the utility file where commonly used helper
functions are stored for our Point-Scattering 
Problem implementations.
"""
# Standard libraries or third-party packages
from dataclasses import dataclass, asdict   # Used for the GA parameters
from typing import Optional

# Local Imports
from implementations import cartesian, polar, boundary

# For calculations
import math

# Plots
import matplotlib.pyplot as plt

# DEAP and helpers
from deap import base, creator, tools

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

            d = math.sqrt((dx_squared) + (dy_squared))   # Replace with Euclidean Distance Formula

            if d < min_dist:
                min_dist = d

            return min_dist

# Formula: (r*cos(theta), r*sin(theta))
def polar_to_cart(ind):
    return [(r*math.cos(theta), r*math.sin(theta)) for r, theta in ind]

# Formula for restricted polar (r=1): (cos(theta), sin(theta)), same formula as above but only 1 input variable (theta)
def angle_to_cart(ind):
    return [(math.cos(theta), math.sin(theta)) for theta in ind]

# ===================== Plotting =====================

# ===================== GA Operators =====================
# Crossover function can be the same across all 3 representations
# Since it is simply a tuple list of floats 
"""
Options:
1. Simple averaging new_X = (X1 + X2)/2, new_Y = (Y1 + Y2)
2. Whole Arithmetic recombination based off Pg.80 from the book
3. Blend crossover
"""

# ===================== GA Configuration Dataclass =====================
# Configuration dataclass
@dataclass(frozen=True)         # Parameters cant be changed during runs
class GAConfig:
    pop_size: int = 200         # 200 individuals
    generations: int = 200      # Allowed generations
    cxpb: float = 0.7           # # crossover prob
    mutpb: float = 0.2          # mutation probability
    tournsize: int = 3          # For tournament slection
    seed: Optional[int] = None

# ===================== Running the Point-Scattering GA =====================
class PointScatteringGA:
    def __init__(self, n: int, representation: str="cartesian", cfg: GAConfig=GAConfig()):
        self.n = n
        self.representation = representation
        self.cfg: GAConfig = cfg

        # Build the DEAP toolbox

    def build_toolbox(self):
        self.ensure_deap_creators()

        ## TOOLBOX ##
        toolbox = base.Toolbox()

        # Operator registration
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        # cxTwoPoint choosen as it mixes parents more evenly
        toolbox.register("mate", tools.cxTwoPoint)      
        toolbox.register("select", tools.selTournament, tournsize=self.cfg.tournsize)   # Was the best selection method from the prev HW

        # The 3 representations will be handled here
        if self.representation == "cartesian":
            # Setup the cartesian-specific operators
            toolbox.register("individual", tools.initIterate, creator.Individual, lambda: cartesian.init_cartesian_ind(self.n))
            toolbox.register("evaluate", cartesian.evalCartesianFitness)
            toolbox.register("mutate", cartesian.mutate_cartesian_ind)

        #elif self.representation == "polar":

        #elif self.representation == "angle":

        else:
            raise ValueError(f"Unknown representation: {self.representation}")

        self.toolbox = toolbox
    
    # Method which ensure that no duplicates are existing
    def ensure_deap_creators(self) -> None:
        if not hasattr(creator, "FitnessMax"):      # Ensure doesnt exist for both FitnessMax & Individual
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
    
    def runGA(self):
        return
