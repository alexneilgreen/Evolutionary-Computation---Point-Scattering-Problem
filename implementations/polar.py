#!/usr/bin/env python3
"""
This is the polar implementation for the Point-Scattering 
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
def init_polar_ind(n):
    """
    Creates our collection of individual points with the polar representation.
    Ensuring valid points within the unit circle range (r = 0 to 1)
    """
    # individual = full set of n points
    # Of form [(r, theta),...,(r_n, theta_n)]
    ind = []

    # Fill the list with n valid individuals
    while len(ind) < n:
        # Assign random coorindates for r between 0 and 1
        r = random.uniform(0, 1)

        # Assign random coorindates for r between 0 and 2*pi
        theta = random.uniform(0, 2*math.pi)

        # Append to coordinates to our ind list
        ind.append((r, theta))
    return ind

# Mutate the current population
def mutate_polar_ind(ind, indpb=0.05):
    """
    ind: Individual of cart representatiom
    indpb: Individual's probability of experiencing mutation
    """
    for mutant in range(len(ind)):
        # Should we mutate
        if random.random() < indpb:
            # Mutate by picking random values
            # Must ensure it is within the unit circle
            r = random.uniform(0, 1)
            theta = random.uniform(0, 2*math.pi)

            ind[mutant] = (r, theta)    # Assign the mutant in ind list its new r and theta
    
    # Return the newly mutated individuals as a tuple
    return (ind,)

# Polar Implementation
def run():
    """Runs the cartessian implementation."""