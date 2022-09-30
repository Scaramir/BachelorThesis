'''
This script will generate two sets of toy data. One with more and one with less dots.
author: Maximilian Otto, 2022, maxotto45@gmail.com
Generate some matrices. The numbers in it represent the size of each dot 
Pseudocode:
[x] 1. Initialize matrix with zeros
[x] 2. for each cell in matrix:
[x]    2. Assign value through random function with given probaility (e.g. 0.3 vs. 0.5)
[x]    3. Value = random[ 0, 1] with given probability (e.g. 0.1)
[x]       if value == 1:
[x]          value = [ 0 : (dist_to_nearest_neighbor) - value.nearest_neighbor]
[x] - plot the matrix (matplotlib). The numvber in each cell is the size of the dot to plot 
[ ] - TODO: Keep the Distances (+sizes) of each dot to another in a list and save teh list for all images while generating. 
        - This will be used to compare the mean distance and shortest path of teh two datasets (e.g. for the distance metric and to find a statistical significant difference)
        - further, this statistic will be used to compare the obtrained ground-truth with what the network learned.
'''

import random
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

# Change directory to the one containing this script
pic_folder_path = "S:/BA/data/toy_data"
dataset_size = 1000
mat_size = 224 
prob_set = 0.00005 #0.005 #0.0002
dot_size_factor = 1 #0.2 #0.8


os.chdir(pic_folder_path)
# Set seeds for reproducibility
random.seed(50318431)
np.random.seed(50318431)

def setup_matrix(mat_size):
    '''
    Initialize matrix of given size² with zeros.
    Returns a numpy array.
    '''
    import numpy as np
    return np.zeros((mat_size, mat_size), dtype=int)

def find_max_space_for_current_point(dot_list, x_coordinate, y_coordinate, mat_size = 224):
    '''
    Find the smallest distance to any dot in a list of given coordinates.
    Return the distance (max space to the border of all neighbors or the matrix walls; and the distance to the nearest neighbor dot).
    The border is the already covered area (each dot has an assigned value, basically its radius). 
    This way, we don't have overlaps.
    :param x_coordinate and 
    :param y_coordinate are the coordinates of the current point. 
    :param mat_size is the size of the matrix.
    These params must not be negative.
    To calculate the distance, a Jump Distance is computed:
    '''
    # minimal distance to the border/wall of the matrix
    max_space_to_border = min(x_coordinate+1, y_coordinate+1, mat_size-1 - x_coordinate + 1, mat_size-1 - y_coordinate + 1)

    # no neighbor dots?
    if len(dot_list) == 0:
        return max_space_to_border, 0

    distances = []
    for dot in dot_list:
        distances.append(np.sqrt((dot[0] - x_coordinate)**2 + (dot[1] - y_coordinate)**2))
    
    # Are all distances >= dot_list[:][2]?
    dist_to_nearest_neighbor_edge = min(distances[i] - dot_list[i][2] + 1 for i in range(len(distances)))

    if max_space_to_border < dist_to_nearest_neighbor_edge:
        return max_space_to_border, min(distances)

    if dist_to_nearest_neighbor_edge <= 0:
        return 0, 0
    else:
        return dist_to_nearest_neighbor_edge, min(distances)

def generate_matrix(matrix, prob_set = 0.1, dot_size_factor = 1):
    '''
    :param matrix = numpy array, 
    :param prob_set = float, 
    :param dot_size_factor = float. 
    prob_set is the probability that a location/cell within the matrix will be considered as to set a dot or not.
    dot_size_factor is the factor by which the size of the dots is multiplied. This will be applied after calculating the distance to the covered border. -> Values above 1 lead to overlapping areas.
    Returns: matrix with dots.

    1. for each cell in matrix:
    2. Randmoly (with a given probability) assign a value to it, or skip the whole cell.
    3. if value gets assigned:
            value = randint(0, (max_space_until_the_already_covered_area)])
            (This step often choose a random integer between 0 and 1)
    '''
    # Matrix is a square
    mat_size = matrix.shape[0]

    dot_list = []
    nearest_neighbor_distances = []

    # For each cell in matrix
    for i in range(0, mat_size):
        for j in range(0, mat_size):
            # Assign value through random function with given probaility (e.g. 0.3 vs. 0.5)
            curr_cell_set = np.random.choice([True, False], p=[prob_set, 1-prob_set])
            if curr_cell_set:
                max_space_to_an_edge, nearest_neighbor_distance = find_max_space_for_current_point(dot_list, i, j, mat_size)
                # Spot is covered by the size/value of a neighbor?
                if int(max_space_to_an_edge) == 0:
                    continue

                matrix[i][j] = np.random.randint(0, max_space_to_an_edge) * dot_size_factor

                if matrix[i][j] != 0:
                    dot_list.append([i, j, matrix[i][j]])
                    nearest_neighbor_distances.append(nearest_neighbor_distance)

    if len(nearest_neighbor_distances) == 0:
        return matrix, dot_list, 0
    nearest_neighbor_distances[nearest_neighbor_distances == 0] = np.nan
    return matrix, dot_list, np.nanmean(nearest_neighbor_distances)

def plot_dots(dot_list, prob_set, dot_size_factor, dir, mat_size, im_count = 1, save_plot = True, show_plot = False):	
    '''
    Input: dot_list = [[x, y, size], [x, y, size], ...]. A list, containing lists of dot-information (coodinates and radius represent a dot).
    Output: Plot of the dots, keeping the position and radius of the dots.
    Effect: Plot gets saved in the given directory.
    '''
    import matplotlib.pyplot as plt

    plt.clf()
    plt.figure(figsize=(2.9, 2.97))
    plt.axis([0, mat_size, 0, mat_size])
    plt.axis("equal")
    plt.axis('off')
    ax = plt.gca()
    fig = plt.gcf()

    for dot in dot_list:
        c = plt.Circle((dot[0]+0.5, dot[1]+0.5),  color='r', radius=dot[2]-0.5, alpha=0.7)
        ax.add_patch(c)

    # save figure as png file in the given directory with a resolution of 224x224 pixels
    if save_plot:
        fig.savefig(dir + "/" + "dot_plot_" + str(prob_set) + "_" + str(dot_size_factor) + "_" + str(im_count) + ".png", dpi=100*(mat_size/224), bbox_inches='tight', pad_inches=0)
    if show_plot:
        plt.show()
    else: 
        plt.close()
    return


def generate_valid_matrix(mat, prob_set, dot_size_factor):
    '''
    Recursive way to generate matrices until one is not completely empty.
    '''
    mat, dot_list, mean_nearest_neighbor_distance = generate_matrix(mat, prob_set, dot_size_factor)
    if not dot_list:
        mat, dot_list, mean_nearest_neighbor_distance = generate_valid_matrix(mat, prob_set, dot_size_factor)        
    return mat, dot_list, mean_nearest_neighbor_distance

def generate_dataset(pic_folder_path, mat_size, prob_set, dot_size_factor, dataset_size = 500):
    '''
    Generates a given amount of images with a given size, with red dots in . 
    The higher the probability, the more dots can be placed. The larger the 
    '''

    processed_files = []
    mean_nearest_neighbor_distances = []

    for im_count in tqdm(range(1, dataset_size + 1)):
        mat = setup_matrix(mat_size)
        mat, dot_list, mean_nearest_neighbor_distance = generate_valid_matrix(mat, prob_set, dot_size_factor)
        # Create the images
        plot_dots(dot_list, prob_set, dot_size_factor, pic_folder_path, mat_size, im_count=im_count, save_plot=True)
        processed_files.append(im_count)
        mean_nearest_neighbor_distances.append(mean_nearest_neighbor_distance)
    # Add processed files and mean nearest neighbor distance to a df
    df = pd.DataFrame({'file': processed_files, 'mean_nearest_neighbor_distance': mean_nearest_neighbor_distances})
    # Save df to a csv file
    df.to_csv(pic_folder_path + "/" + "dataset_" + str(prob_set) + "_" + str(dot_size_factor) + "_mean_distance_per_image.csv", index=False)

    return df

# --------------------------------- MAIN --------------------------------- #
df = generate_dataset(pic_folder_path, mat_size, prob_set, dot_size_factor, dataset_size)