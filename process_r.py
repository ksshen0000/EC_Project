import numpy as np
import pygad
import cv2
row = 100
col = 100
alpha = 0.8

def split_RGBThreeChannel(img):
    img = img / 255.0
    (B, G, R) = cv2.split(img) # 3 channel
    b_channel = B.flatten()
    g_channel = G.flatten()
    r_channel = R.flatten()

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

    fitness = np.sum(np.abs(r_channel - solution))
    fitness = 10000 - fitness
    return fitness
    
    

def callback_R(ga_instance):
    # print("Generation = {gen}".format(gen=ga_instance.generations_completed))
    # print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    # print("R")

    if ga_instance.generations_completed % 500 == 0:
        zeros = np.zeros(row*col, dtype = np.uint8)
        cv2.imwrite('./R_sol/R_solution_'+str(ga_instance.generations_completed)+'.png', merge_RGBThreeChannel(ga_instance.best_solution()[0], zeros, zeros, row, col))

def crossover_single_arithmetic(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    
    # alpha = np.random.uniform(0.2, 0.8, 1)
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        child = parent1*alpha + parent2*(1-alpha)
        offspring.append(child)

        idx += 1

    return np.array(offspring)

def process_ga_r():

    image = cv2.imread("ea_test.png")
    global r_channel
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
                       on_generation=callback_R,
                       crossover_type=crossover_single_arithmetic)
    ga_instance_r.run()
    ga_instance_r.save(filename="ga_instance_r")
    (solution_r, solution_fitness_r, solution_idx_r) = ga_instance_r.best_solution()
    # print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness_r))
    # print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx_r))
    with open('r.npy', "wb") as f:
        np.save(f, solution_r)
    print("r save")
    