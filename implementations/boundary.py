#!/usr/bin/env python3
"""
This is the custom boundary implementation for the Point-Scattering 
Problem.
"""
# Standard libraries or third-party packages
import random
import math
from deap import base, creator, tools

# Local Imports
import utility

# Will need to handle for this specific representation
# 1. Generating population of our individuals points 
# 2. Evaluating Fitness level i.e maximum minimum-distance between point pairs
# 3. Mutating the individuals



# Create n points within circle
def init_boundary_ind(n):
    """
    Creates our collection of individual points with the boundary representation.
    Ensuring valid points on the unit circle boundary (r = 1)
    """
    # individual = full set of n points
    # Of form [(1, theta),...,(1, theta_n)]
    ind = []

    # Fill the list with n valid individuals
    while len(ind) < n:
        # Assign random coorindates for r between 0 and 2*pi
        theta = random.uniform(0, 2*math.pi)

        # Append to coordinates to our ind list
        ind.append((theta))
    return ind

# Mutate the current population
def mutate_boundary_ind(ind, indpb=0.05):
    """
    ind: Individual of cart representatiom
    indpb: Individual's probability of experiencing mutation
    """
    for mutant in range(len(ind)):
        # Should we mutate
        if random.random() < indpb:
            # Mutate by picking random values
            theta = random.uniform(0, 2*math.pi)

            ind[mutant] = (theta)    # Assign the mutant in ind list its new theta
    
    # Return the newly mutated individuals as a tuple
    return (ind,)

# Boundary Implementation
def run():
    """Runs the cartessian implementation."""