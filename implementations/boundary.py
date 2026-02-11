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
        r = 1       # Always this, because we are 1D, caring only about the angle
        # Assign random coorindates for theta between 0 and 2*pi
        theta = random.uniform(0, 2*math.pi) 

        # Append to coordinates to our ind list
        ind.append((r, theta))
    return ind

# Mutate the current population
def mutate_boundary_ind(ind, indpb=0.2):
    """
    ind: Individual of cart representatiom
    indpb: Individual's probability of experiencing mutation
    """
    for mutant in range(len(ind)):
        # Should we mutate
        if random.random() < indpb:
            # Mutate by picking random values
            theta = random.uniform(0, 2*math.pi)                    # Additional mutation op: + random.gauss(0, 0.2) % (2 * math.pi)

            ind[mutant] = (1, theta)    # Assign the mutant in ind list its new theta
    
    # Return the newly mutated individuals as a tuple
    return (ind,)

# Boundary Implementation
def run(args):

    # Use Standard Config from Utitilty
    cfg = utility.Config()

    # DEAP creator setup
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    # Setup toolbox
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, 
                     lambda: init_boundary_ind(args.n))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", utility.calcMinEuclideanDistancePolar)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)  # Uniform crossover
    toolbox.register("mutate", mutate_boundary_ind, indpb=args.indpb)
    toolbox.register("select", tools.selTournament, tournsize=cfg.tournsize)

    # Create initial popultation
    population = toolbox.population(n=cfg.pop_size)

    # Plot initial point locations
    initial_individual_cartesian = utility.polar_to_cart(population[0])   # Convert to cartesian before plotting
    utility.plot_point_distribution(initial_individual_cartesian, title=f"Initial Population (n={args.n})",
        filename=f"boundary_n{args.n}_initial.png")

    # Evaluate initial population
    fitnesses = map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Track Performance of Generations
    log = []

    # Open log file
    log_filename = f"logs/boundary_n{args.n}_gen{cfg.generations}.txt"

    with open(log_filename, 'w') as log_file:
        log_file.write(f"Boundary Representation Log\n")
        log_file.write(f"n={args.n}, generations={cfg.generations}, population={cfg.pop_size}\n")
        log_file.write(f"crossover_prob={cfg.cxpb}, mutation_prob={cfg.mutpb}, indpb={args.indpb}, seed={args.seed}\n")
        log_file.write("=" * 80 + "\n\n")
        
        # Evolution loop
        for gen in range(cfg.generations):
            # Select offspring
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))
            
            # Apply crossover
            for i in range(0, len(offspring), 2):
                if i + 1 < len(offspring):
                    child1 = offspring[i]
                    child2 = offspring[i + 1]
                
                    if random.random() < cfg.cxpb:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
            
            # Apply mutation
            for mutant in offspring:
                if random.random() < cfg.mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate individuals with invalid fitness
            invalid_ind = []
            for ind in offspring:
                if not ind.fitness.valid:   # Check Fitness
                    invalid_ind.append(ind)

            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace population
            for i in range(len(population)):
                population[i] = offspring[i]
            
            # Get best individual of this generation
            best_ind = tools.selBest(population, 1)[0]
            best_fitness = best_ind.fitness.values[0]
            
            # Record performance
            log.append((gen, best_fitness))
            
            # Convert to Cartesian and Log this generation
            best_ind_cart = utility.polar_to_cart(best_ind)
            utility.log_generation(log_file, gen, best_ind_cart)
    
    # Final best solution
    best_individual = tools.selBest(population, 1)[0]
    best_fitness = best_individual.fitness.values[0]

    # Plot final point locations
    final_individual_cartesian = utility.polar_to_cart(population[0])   # Convert to cartesian before plotting
    utility.plot_point_distribution(final_individual_cartesian, title=f"Final Population (n={args.n})",
        filename=f"boundary_n{args.n}_final.png")
    
    print(f"\tFinal Best Minimum Distance: {best_fitness:.6f}\n")
    
    # Plot results
    title = f"Boundary Representation (n={args.n})"
    filename = f"boundary_n{args.n}_gen{cfg.generations}.png"
    utility.plot_fitness_log(log, title, filename)
