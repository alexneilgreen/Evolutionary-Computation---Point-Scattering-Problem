#!/usr/bin/env python3
"""
This is the cartessian implementation for the Point-Scattering 
Problem.
"""
# Standard libraries or third-party packages

# Local Imports
import random
import utility

def init_cartesian_ind(n):
    """
    Creates our collection of individual points with the cartesian representation.
    Ensuring valid points within the unit circle range
    """
    # individual = full set of n points
    # Of form [(X,Y),...,(Xn,Yn)]
    ind = []

    # Fill the list with n valid individuals
    while len(ind) < n:
        # Assign random coorindates for X and Y, between -1 and 1
        x, y = random.uniform(-1, 1), random.uniform(-1, 1)

        # Check in the unit circle
        if utility.in_unitCircle(x, y):
            # we can append to coordinates to our ind list
            ind.append((x,y))   # ind + 1
    return ind

# To Evalulate fitness, simply calc the minimum Euclidean distance
def evalCartesianFitness(individuals):
    return (utility.calcMinEuclideanDistance(individuals))

def mutate_cartesian_ind(ind, indpb=0.05):
    """
    ind = individual of cart representatiom
    indpb: Individual's probability of experiencing mutation
    """
    for mutant in range(len(ind)):
        # Should we mutate
        if random.random() < indpb:
            # Mutate:By picking random values
            # Must ensure it is within the unit circle
            while True:
                x, y = random.uniform(-1, 1), random.uniform(-1, 1)

                # Check in the unit circle
                if utility.in_unitCircle(x, y):
                    ind[mutant] = (x, y)        # Assign the mutant in ind list its new X and Y
                    break 
    
    # Return the newly mutated individuals as a tuple
    return (ind,)