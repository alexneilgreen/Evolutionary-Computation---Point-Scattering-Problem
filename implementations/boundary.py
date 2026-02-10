#!/usr/bin/env python3
"""
This is the custom boundary implementation for the Point-Scattering 
Problem.
"""
# Standard libraries or third-party packages
import math
import random
from deap import base, creator, tools

# Local Imports
import utility

# Will need to handle for this specific representation
# 1. Generating population of our individuals points 
# 2. Evaluating Fitness level i.e maximum minimum-distance between point pairs
# 3. Mutating the individuals

def init_boundary_ind(n):
    ind = []

    for i in range(n):
        # Create individuals between Angles 0 <= theta <= 360 
        theta = random.uniform(0, 2*math.pi)
        ind.append(theta)

    return ind

# Mutate the current population
# Since we have 
def mutate_boundary_ind(ind, indpb=0.05):
    return

def run():
    """Runs the cartessian implementation."""