#!/usr/bin/env python3
"""
This file calls the individual implementations for the Point-Scattering 
Problem.

Each implementation has it's own file in order to setup DEAP framework 
independently of each other. Common functions that are used in more 
than one implementation are stored within the utility.py file.
"""

"""
Resources:
DEAP Setup - https://deap.readthedocs.io/en/master/examples/ga_onemax.html
DEAP Functions - https://deap.readthedocs.io/en/master/api/tools.html
"""

# Standard libraries or third-party packages
import argparse
import os
import random
import numpy as np

# Local Imports
from implementations import cartesian, polar, boundary

def setup_directories():
    """Create necessary directories for outputs"""
    os.makedirs('graphs', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

def main():
    """Main function of the script."""
    parser = argparse.ArgumentParser(
        description='Point Scattering Problem - GA Comparison of Three Representations'
    )

    parser.add_argument('--n', type=int, default=5,                     # Set third custom n value
                        help='Number of points to place (default: 5)')
    parser.add_argument('--indpb', type=float, default=0.2,             # Set indpb value
                        help='Independent probability for mutating each gene (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,                 # Set seed value (shouldn't change)
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Create output directories
    setup_directories()
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Run Cartesian Implementation
    print("===== Cartesian Implementation (x, y) =====\n")
    cartesian.run(args)
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Run Polar Implementation
    print("===== Polar Implementation (r, θ) =====\n")
    polar.run(args)

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Run Custom Boundary Implementation
    print("===== Boundary Implementation (θ) =====\n")
    boundary.run(args)

if __name__ == "__main__":
    main()
