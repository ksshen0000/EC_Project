import numpy as np
import pygad
import cv2
row = 100
col = 100
alpha = 0.2

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
  
def fitness_G(solution, solution_idx):

    fitness = np.sum(np.abs(g_channel - solution))
    fitness = 10000 - fitness
    return fitness
    
    

def callback_G(ga_instance):
    # print("Generation = {gen}".format(gen=ga_instance.generations_completed))
    # print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    # print("R")

    if ga_instance.generations_completed % 500 == 0:
        zeros = np.zeros(row*col, dtype = np.uint8)
        cv2.imwrite('./G_sol/G_solution_'+str(ga_instance.generations_completed)+'.png', merge_RGBThreeChannel(zeros, ga_instance.best_solution()[0], zeros, row, col))

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

def crossover_blend(parents, offspring_size, ga_instance):
    offspring = []
    idx=0
    while len(offspring) != offspring_size[0]:
        parent1 = np.array(parents[idx % parents.shape[0], :].copy())
        parent2 = np.array(parents[(idx + 1) % parents.shape[0], :].copy())
        child_lb = parent1 - alpha*(parent2-parent1)
        child_up = parent2 + alpha*(parent2-parent1)
        
        child = np.random.uniform(child_lb,child_up)
        child.clip(0,1)
        offspring.append(child)
        idx+=1

    return np.array(offspring)

def crossover_linear_g(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    # alpha = np.random.uniform(0.2, 0.8, 1)
    while len(offspring) != offspring_size[0]:
        parent1 = np.array( parents[idx % parents.shape[0], :].copy())
        parent2 = np.array(parents[(idx + 1) % parents.shape[0], :].copy())
        arena=[]
        fitness_arena=[]
        arena.append(0.5*parent1+0.5*parent2)
        sol_temp = 1.5*parent1-0.5*parent2
        sol_temp[sol_temp>1.0] = 1.0
        sol_temp[sol_temp<0.0] = 0.0
        arena.append(sol_temp)
        sol_temp_2 = -0.5*parent1+1.5*parent2
        sol_temp_2[sol_temp_2>1.0] = 1.0
        sol_temp_2[sol_temp_2<0.0] = 0.0
        arena.append(sol_temp_2)
        maxi=0
        for i in range (0,3):
            fitness_arena.append(fitness_G(arena[i],i))
            if ( fitness_arena[maxi] <fitness_arena[i]):
                maxi = i
        offspring.append(arena[maxi])

        idx += 1

    return np.array(offspring)

def process_ga_g():

    image = cv2.imread("ea_test.png")
    global g_channel
    (r_channel, g_channel, b_channel) = split_RGBThreeChannel(image)
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
                       on_generation=callback_G,
                       crossover_type=crossover_blend,
                       parent_selection_type="tournament")
    ga_instance_g.run()
    ga_instance_g.save(filename="ga_instance_g")
    (solution_g, solution_fitness_g, solution_idx_g) = ga_instance_g.best_solution()
    print("G : Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness_g))
    # print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx_g))
    with open('g.npy', "wb") as f:
        np.save(f, solution_g)
    print("G save")
    