#!/usr/bin/env python3
"""
This is the cartessian implementation for the Point-Scattering 
Problem.
"""
# Standard libraries or third-party packages
import random
from deap import base, creator, tools

# Local Imports
import utility



# Create n points within circle
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
            ind.append((x, y))   # ind + 1
    return ind

# Mutate the current population
def mutate_cartesian_ind(ind, indpb=0.2):
    """
    ind: Individual of cart representatiom
    indpb: Individual's probability of experiencing mutation
    """
    for mutant in range(len(ind)):
        # Should we mutate
        if random.random() < indpb:
            # Mutate by picking random values
            # Must ensure it is within the unit circle
            while True:
                x, y = random.uniform(-1, 1), random.uniform(-1, 1)

                # Check in the unit circle
                if utility.in_unitCircle(x, y):
                    ind[mutant] = (x, y)        # Assign the mutant in ind list its new X and Y
                    break 
    
    # Return the newly mutated individuals as a tuple
    return (ind,)

# Cartesian Implementation
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
                     lambda: init_cartesian_ind(args.n))                            # links and creates individuals using custom function
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", utility.calcMinEuclideanDistance)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)                            # Uniform crossover
    toolbox.register("mutate", mutate_cartesian_ind, indpb=args.indpb)
    toolbox.register("select", tools.selTournament, tournsize=cfg.tournsize)

    # Create initial popultation
    population = toolbox.population(n=cfg.pop_size)

    # Plot initial point locations
    utility.plot_point_distribution(population[0], title=f"Initial Population (n={args.n})",
        filename=f"cartesian_n{args.n}_initial.png")

    # Evaluate initial population
    fitnesses = map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Track Performance of Generations
    log = []

    # Open log file
    log_filename = f"logs/cartesian_n{args.n}_gen{cfg.generations}.txt"

    with open(log_filename, 'w') as log_file:
        log_file.write(f"Cartesian Representation Log\n")
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
            
            # Log this generation
            utility.log_generation(log_file, gen, best_ind)
    
    # Final best solution
    best_individual = tools.selBest(population, 1)[0]
    best_fitness = best_individual.fitness.values[0]

    # Plot final point locations
    utility.plot_point_distribution(population[0], title=f"Final Population (n={args.n})",
        filename=f"cartesian_n{args.n}_final.png")
    
    print(f"\tFinal Best Minimum Distance: {best_fitness:.6f}\n")
    
    # Plot results
    title = f"Cartesian Representation (n={args.n})"
    filename = f"cartesian_n{args.n}_gen{cfg.generations}.png"
    utility.plot_fitness_log(log, title, filename)