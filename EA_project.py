import cv2
import numpy as np
import matplotlib.pyplot as plt
import pygad

r_channel = []
g_channel = []
b_channel = []
row = 100
col = 100
def split_RGBThreeChannel(img):
    img = img / 255.0
    (B, G, R) = cv2.split(img) # 3 channel
    b_channel = B.flatten()
    g_channel = G.flatten()
    r_channel = R.flatten()
    # make all zeros channel
    # zeros = np.zeros(img.shape[:2], dtype = np.uint8)
    # cv2.imshow('R', merge_RGBThreeChannel(R=R, G=zeros, B=zeros))
    # cv2.waitKey(0)
    
    # cv2.imshow('G', merge_RGBThreeChannel(R=zeros, G=G, B=zeros))
    # cv2.waitKey(0)
    
    # cv2.imshow('B', merge_RGBThreeChannel(R=zeros, G=zeros, B=B))
    # cv2.waitKey(0)

    return (r_channel, g_channel, b_channel)

def merge_RGBThreeChannel(R, G, B, row, col):
    r_channel = R.reshape(row, col, 1)
    g_channel = G.reshape(row, col, 1)
    b_channel = B.reshape(row, col, 1)
    r_channel = r_channel.astype(float)
    g_channel = g_channel.astype(float)
    b_channel = b_channel.astype(float)
    img = cv2.merge([b_channel, g_channel, r_channel])
    img = img * 255.0
    return img

def fitness_R(solution, solution_idx):
    """
    Calculating the fitness value for a solution in the population.
    The fitness value is calculated using the sum of absolute difference between genes values in the original and reproduced chromosomes.
    
    solution: Current solution in the population to calculate its fitness.
    solution_idx: Index of the solution within the population.
    """

    fitness = np.sum(np.abs(r_channel - solution))

    # Negating the fitness value to make it increasing rather than decreasing.
    # fitness = np.sum(r_channel) - fitness
    fitness = 10000 - fitness
    return fitness

def fitness_G(solution, solution_idx):
    """
    Calculating the fitness value for a solution in the population.
    The fitness value is calculated using the sum of absolute difference between genes values in the original and reproduced chromosomes.
    
    solution: Current solution in the population to calculate its fitness.
    solution_idx: Index of the solution within the population.
    """

    fitness = np.sum(np.abs(g_channel - solution))

    # Negating the fitness value to make it increasing rather than decreasing.
    # fitness = np.sum(r_channel) - fitness
    fitness = 10000 - fitness
    return fitness

def fitness_B(solution, solution_idx):
    """
    Calculating the fitness value for a solution in the population.
    The fitness value is calculated using the sum of absolute difference between genes values in the original and reproduced chromosomes.
    
    solution: Current solution in the population to calculate its fitness.
    solution_idx: Index of the solution within the population.
    """

    fitness = np.sum(np.abs(b_channel - solution))

    # Negating the fitness value to make it increasing rather than decreasing.
    # fitness = np.sum(r_channel) - fitness
    fitness = 10000 - fitness
    return fitness

def callback_R(ga_instance):
    # print("Generation = {gen}".format(gen=ga_instance.generations_completed))
    # print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

    if ga_instance.generations_completed % 500 == 0:
        zeros = np.zeros(row*col, dtype = np.uint8)
        cv2.imwrite('./R_sol/R_solution_'+str(ga_instance.generations_completed)+'.png', merge_RGBThreeChannel(ga_instance.best_solution()[0], zeros, zeros, row, col))

def callback_G(ga_instance):
    # print("Generation = {gen}".format(gen=ga_instance.generations_completed))
    # print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

    if ga_instance.generations_completed % 500 == 0:
        zeros = np.zeros(row*col, dtype = np.uint8)
        cv2.imwrite('./G_sol/G_solution_'+str(ga_instance.generations_completed)+'.png', merge_RGBThreeChannel(zeros, ga_instance.best_solution()[0], zeros, row, col))

def callback_B(ga_instance):
    # print("Generation = {gen}".format(gen=ga_instance.generations_completed))
    # print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

    if ga_instance.generations_completed % 500 == 0:
        zeros = np.zeros(row*col, dtype = np.uint8)
        cv2.imwrite('./B_sol/B_solution_'+str(ga_instance.generations_completed)+'.png', merge_RGBThreeChannel(zeros, zeros, ga_instance.best_solution()[0], row, col))


if __name__=='__main__':
    image = cv2.imread("ea_test.png")
    (r_channel, g_channel, b_channel) = split_RGBThreeChannel(image)

    ga_instance_r = pygad.GA(num_generations=20000,
                       num_parents_mating=10,
                       fitness_func=fitness_R,
                       sol_per_pop=20,
                       num_genes=row*col,
                       init_range_low=0.0,
                       init_range_high=1.0,
                       mutation_percent_genes=0.05,
                       mutation_type="random",
                       mutation_by_replacement=True,
                       random_mutation_min_val=0.0,
                       random_mutation_max_val=1.0,
                       on_generation=callback_R)

    ga_instance_g = pygad.GA(num_generations=20000,
                       num_parents_mating=10,
                       fitness_func=fitness_G,
                       sol_per_pop=20,
                       num_genes=row*col,
                       init_range_low=0.0,
                       init_range_high=1.0,
                       mutation_percent_genes=0.05,
                       mutation_type="random",
                       mutation_by_replacement=True,
                       random_mutation_min_val=0.0,
                       random_mutation_max_val=1.0,
                       on_generation=callback_G)

    ga_instance_b = pygad.GA(num_generations=20000,
                       num_parents_mating=10,
                       fitness_func=fitness_B,
                       sol_per_pop=20,
                       num_genes=row*col,
                       init_range_low=0.0,
                       init_range_high=1.0,
                       mutation_percent_genes=0.05,
                       mutation_type="random",
                       mutation_by_replacement=True,
                       random_mutation_min_val=0.0,
                       random_mutation_max_val=1.0,
                       on_generation=callback_B)
    
    ga_instance_r.run()
    ga_instance_b.run()
    ga_instance_g.run()

    # After the generations complete, some plots are showed that summarize the how the outputs/fitenss values evolve over generations.
    # ga_instance_r.plot_fitness()

    # Returning the details of the best solution.
    (solution_r, solution_fitness_r, solution_idx_r) = ga_instance_r.best_solution()
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness_r))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx_r))

    (solution_g, solution_fitness_g, solution_idx_g) = ga_instance_g.best_solution()
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness_g))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx_g))

    (solution_b, solution_fitness_b, solution_idx_b) = ga_instance_b.best_solution()
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness_b))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx_b))

    # if ga_instance_r.best_solution_generation != -1:
    #     print("Best fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance_r.best_solution_generation))
        
    result = merge_RGBThreeChannel(solution_r, solution_g, solution_b, row, col)
    
    cv2.imwrite("ea_result.png", result)
    result_dis = cv2.imread("ea_result.png")
    plt.imshow(cv2.cvtColor(result_dis, cv2.COLOR_BGR2RGB))
    plt.title("EA project : Reproducing Images by GA")
    plt.show()
    
   
