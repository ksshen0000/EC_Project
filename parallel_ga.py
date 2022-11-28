import cv2
import numpy as np
import matplotlib.pyplot as plt
import pygad
import multiprocessing as mp
import time
from process_r import process_ga_r
from process_g import process_ga_g
from process_b import process_ga_b
row = 100
col = 100


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

if __name__=='__main__':

    image = cv2.imread("ea_test.png")
    global r_channel, g_channel, b_channel
    (r_channel, g_channel, b_channel) = split_RGBThreeChannel(image)
    start = time.time()
    p1 = mp.Process(target=process_ga_r, args=())
    p2 = mp.Process(target=process_ga_g, args=())
    p3 = mp.Process(target=process_ga_b, args=())
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()
    
    
    ga_instance_r = pygad.load(filename="ga_instance_r")
    ga_instance_g = pygad.load(filename="ga_instance_g")
    ga_instance_b = pygad.load(filename="ga_instance_b")
    
    # ga_instance_r.plot_fitness()
    # ga_instance_g.plot_fitness()
    with open('r.npy', 'rb') as f:
        solution_r = np.load(f)
    with open('g.npy', 'rb') as f:
        solution_g = np.load(f)
    with open('b.npy', 'rb') as f:
        solution_b = np.load(f)
    end = time.time()
    print("The time => " + str(end - start))
    
    result = merge_RGBThreeChannel(solution_r, solution_g, solution_b, row, col)
    
    cv2.imwrite("ea_result.png", result)
    result_dis = cv2.imread("ea_result.png")
    plt.imshow(cv2.cvtColor(result_dis, cv2.COLOR_BGR2RGB))
    plt.title("EA project : Reproducing Images by GA")
    plt.show()