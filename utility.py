#!/usr/bin/env python3
"""
This is the utility file where commonly used helper
functions are stored for our Point-Scattering 
Problem implementations.
"""
# Standard libraries or third-party packages
import math

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
    return 

# Formula for restricted polar (r=1): (cos(theta), sin(theta))
def angle_to_cart(ind):
    return