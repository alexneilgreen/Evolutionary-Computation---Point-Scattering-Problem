#!/usr/bin/env python3
"""
This is the polar implementation for the Point-Scattering 
Problem.
"""
# Standard libraries or third-party packages
import math
import random
import numpy as np
from typing import Any, Dict, List
from deap import base, creator, tools
from dataclasses import asdict



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
def mutate_polar_ind(ind, indpb=0.2):
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
def run_single(args):
    # Use Standard Config from Utitilty
    cfg = utility.Config()

    # Each run needs a unique seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # DEAP creator setup
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    # Setup toolbox
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, 
                     lambda: init_polar_ind(args.n))                            # links and creates individuals using custom function
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", utility.calcMinEuclideanDistancePolar)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)                        # Uniform crossover (Book Pg.71)
    toolbox.register("mutate", mutate_polar_ind, indpb=args.indpb)
    toolbox.register("select", tools.selTournament, tournsize=cfg.tournsize)

    # Create initial popultation
    population = toolbox.population(n=cfg.pop_size)

    # Plot initial point locations
    # initial_individual_cartesian = utility.polar_to_cart(population[0])   # Convert to cartesian before plotting
    # utility.plot_point_distribution(initial_individual_cartesian, title=f"Initial Population (n={args.n})",
    #     filename=f"polar_n{args.n}_initial.png")

    # Evaluate initial population
    fitnesses = map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Track Performance of Generations
    log = []

    best_by_gen: List[float] = []
    avg_by_gen: List[float] = []

    def record(gen_idx: int) -> None:
        """ Records the best and avg pop fitness for this gen"""
        fits = [ind.fitness.values[0] for ind in population]
        best = max(fits)
        avg = float(np.mean(fits))

        # Append best and average of each gen to the list
        best_by_gen.append(best)
        avg_by_gen.append(avg)

    record(0)

    # Open log file
    log_filename = f"logs/polar_n{args.n}_gen{cfg.generations}.txt"

    with open(log_filename, 'w') as log_file:
        log_file.write(f"Polar Representation Log\n")
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
            record(gen)
            
            # Convert to Cartesian and Log this generation
            best_ind_cart = utility.polar_to_cart(best_ind)
            utility.log_generation(log_file, gen, best_ind_cart)
    
    # Final best solution
    best_individual = tools.selBest(population, 1)[0]
    best_fitness = best_individual.fitness.values[0]

    # Plot final point locations
    # final_individual_cartesian = utility.polar_to_cart(population[0])   # Convert to cartesian before plotting
    # utility.plot_point_distribution(final_individual_cartesian, title=f"Final Population (n={args.n})",
    #     filename=f"polar_n{args.n}_final.png")
    
    # print(f"\tFinal Best Minimum Distance: {best_fitness:.6f}\n")
    
    # # Plot results
    # title = f"Polar Representation (n={args.n})"
    # filename = f"polar_n{args.n}_gen{cfg.generations}.png"
    # utility.plot_fitness_log(log, title, filename)
    # return info for stat
    return {
        "best_by_gen": best_by_gen,
        "avg_by_gen": avg_by_gen,
        "best_individual": best_individual, # polar still
        "best_overall_fitness": best_fitness,
        "config": asdict(cfg)       # Current GA settings
    }

# Multiple runs of the GA
def run_experiment(args, n_runs: int = 25, seed_base: int = 12345) -> Dict[str, Any]:
    best_by_gen_all = []
    avg_by_gen_all = []
    best_overall_all = []

    # Tracking global best results
    best_run_curve = None
    best_ind = None
    best_ind_fitness = -float("inf")

    for i in range(n_runs):
        # print(f"Run {i}")
        args.seed = seed_base + i   # Creates a unique cfg for each run
        cur_run = run_single(args)

        # Extract the data from current run
        best_by_gen_all.append(cur_run["best_by_gen"])
        avg_by_gen_all.append(cur_run["avg_by_gen"])
        best_overall_all.append(cur_run["best_overall_fitness"])

        # Track the best fitness individual as encountered
        if cur_run["best_overall_fitness"] > best_ind_fitness:
            # Best found is current run
            best_run_curve = cur_run["best_by_gen"]
            best_ind = cur_run["best_individual"]
            best_ind_fitness = cur_run["best_overall_fitness"]
 
    gen_mean, gen_CI_low, gen_CI_high = utility.per_gen_mean_ci(best_by_gen_all)
    mean_f, std_f, CI = utility.mean_std_ci95(best_overall_all)

    # Plot best results
    title = f"Polar Representation (n={args.n})"
    filename = f"polar_n{args.n}_best_run.png"
    utility.plot_fitness_log(list(enumerate(best_run_curve)), title, filename)

    # Polar -> Cart 
    best_ind_cart = utility.polar_to_cart(best_ind)
    # Plot final point locations
    utility.plot_point_distribution(best_ind_cart, title=f"Final Population (n={args.n})",
        filename=f"polar_n{args.n}_best_final.png")

    return {
        "n_runs": n_runs,
        "best_by_gen_all": best_by_gen_all,
        "avg_by_gen_all": avg_by_gen_all,
        "best_overall_all": best_overall_all,
        "final_stats": {
            "mean": mean_f, 
            "std":std_f, 
            "CI95": CI
        },
        "gen_stats": {
            "mean": gen_mean,
            "CI95": (gen_CI_low, gen_CI_high)
        }
    }