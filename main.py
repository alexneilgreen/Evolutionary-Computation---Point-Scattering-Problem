#!/usr/bin/env python3
"""
This file calls the individual implementations for the Point-Scattering 
Problem.

Each implementation has it's own file in order to setup DEAP framework 
independently of each other. Common functions that are used in more 
than one implementation are stored within the utility.py file.
"""
# Standard libraries or third-party packages

import argparse
from email import parser
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
    
    args = parser.parse_args()
    
    # Create output directories
    setup_directories()
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Run Cartesian Implementation
    print("===== Cartesian Implementation (x, y) =====\n")
    cartesian.run(args)
    
    # # Set seed
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    
    # # Run Polar Implementation
    # print("===== Polar Implementation (r, θ) =====\n")
    # polar.run(args)

    # # Set seed
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    
    # # Run Custom Boundary Implementation
    # print("===== Boundary Implementation (θ) =====\n")
    # boundary.run(args)

if __name__ == "__main__":
    main()
