#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pdesrosiers
"""


import numpy as np
from numba import jit
import skimage.io as io
import matplotlib.pyplot as plt
from skimage import exposure, morphology, measure


def infer_structural_graph(cell_mask, neurite_mask, find_paths = False):
    """
    Get a graph model of a neuronal network based on two mask images (matrices),
    one for cell bodies and the other for neurites.

    Arguments
    ---------
    cell_mask       2d numpy array, in which positive integers indicate the pixels
                    that are associated to cell bodies
    neurite_mask    2d numpy array, same size as cell_mask, in which positive entries
                    correspond to pixels belonging to a neurite
    find_paths      boolean, if True will also find the sorthest paths (in pixels)
                    between cells that belong to an edge

    Returns
    -------
    vertices        dict, where each key corresponds to a cell id and its value gives
                    the matrix index (i,j) where the centoid of the cell is.
    weights         2d array, the value at (i,j) is inversely proporttional to the length
                    in pixels of a path along the neurites from cell i to cell j
    paths           list of lists, the latter representing the paths (in neurite_mask)
                    between one cell body to another
    paths_matrix    2d numpy array, same size as cell_mask,
                    if element (i,j) = k, then k shortest paths pass on this element in neurite_mask
    """
    # Tranform the masks into a maze where -1 indicates a wall (background), 0 is a point
    # where paths can pass, and k>0 indicated the centoid of the k-th cell (vertex id k-1)
    # No compression is used
    maze, _ = mask_images_to_maze(cell_mask, neurite_mask, compress_factor = 1, threshold = 0.5)

    # Get the vertices and weights from a fast Numba function
    vertices, weights = infer_graph_from_maze(maze)

    if find_paths:
        paths, paths_matrix = infer_paths_from_maze_vertices_weights(maze, vertices, weights)
        return vertices, weights, paths, paths_matrix

    else:
        return vertices, weights





@jit(nopython=True)
def infer_graph_from_maze(maze):
    """
    Usage
    -----
    mat = np.array([[0,1,0,-1,0,-1],[-1,-1,0,-1,-1,-1],[-1,0,0,0,3,-1],[4,0,0,-1,0,0],[5,0,0,-1,0,2],[-1,-1,0,0,0,0]])
    vertices, weights = infer_graph_from_maze(mat)
    """
    number_vertices = maze.max()
    vertices = dict()
    for i in range(0,number_vertices):
        coord = np.argwhere(maze == i+1) # vertex indices start at 1 in maze
        k = coord[0][0]
        l = coord[0][1]
        vertices[i] = (k,l)

    weight_matrix = np.zeros((number_vertices,number_vertices))
    m, n = maze.shape
    max_iter = m*n
    current_maze = maze
    edge_set = find_edge(current_maze)
    for edge in edge_set:
        weight_matrix[edge[0]-1,edge[1]-1] = 1
    for step in range(1,max_iter):
        new_maze = propagate_sources(current_maze)
        moved = (new_maze != current_maze).sum()>0
        if not moved:
            break
        else:
            current_maze = new_maze
            edge_set = find_edge(current_maze)
            for edge in edge_set:
                if weight_matrix[edge[0]-1,edge[1]-1]==0:
                    weight_matrix[edge[0]-1,edge[1]-1] = 1/(step + 1)

    weight_matrix = weight_matrix + weight_matrix.T
    return vertices, weight_matrix


#@jit(nopython=True)
def infer_paths_from_maze_vertices_weights(maze, vertices, weights):

    # Find edges
    n = weights.shape[0]
    edges = []
    for i in range(n-1):
        for j in range(i,n):
            if weights[i,j]>0:
                edges.append((i,j))

    # Transform ternary maze to binary maze
    binary_maze =np.zeros(maze.shape)
    negative_elements = np.argwhere(maze<0)
    for pair in negative_elements:
        i = pair[0]
        j = pair[1]
        binary_maze[i,j] = 1

    # Find shortest path between pairs of vertives forming edges
    paths = []
    paths_matrix = np.zeros(maze.shape)
    for edge in edges:
        source_id = edge[0]
        target_id = edge[1]
        source = vertices[source_id]
        target = vertices[target_id]
        path, path_matrix = solve_maze(binary_maze, source, target)
        paths.append(path)
        paths_matrix += path_matrix

    return paths, paths_matrix


@jit(nopython=True)
def infer_paths_from_maze(maze):
    """
    Usage
    -----
    mat = np.array([[0,1,0,-1,0,-1],[-1,-1,0,-1,-1,-1],[-1,0,0,0,3,-1],[4,0,0,-1,0,0],[5,0,0,-1,0,2],[-1,-1,0,0,0,0]])
    paths, paths_matrix = infer_paths_from_maze(mat)
    """
    # Find vertices and edges
    vertices, weights = infer_graph_from_maze(maze)
    n = weights.shape[0]
    edges = []
    for i in range(n-1):
        for j in range(i,n):
            if weights[i,j]>0:
                edges.append((i,j))

    # Transform ternary maze to binary maze
    binary_maze =np.zeros(maze.shape)
    negative_elements = np.argwhere(maze<0)
    for pair in negative_elements:
        i = pair[0]
        j = pair[1]
        binary_maze[i,j] = 1

    # Find shortest path between pairs of vertives forming edges
    paths = []
    paths_matrix = np.zeros(maze.shape)
    for edge in edges:
        source_id = edge[0]
        target_id = edge[1]
        source = vertices[source_id]
        target = vertices[target_id]
        path, path_matrix = solve_maze(binary_maze, source, target)
        paths.append(path)
        paths_matrix += path_matrix

    return paths, paths_matrix

def mask_images_to_maze(cell_mask, neurite_mask, compress_factor = 4, threshold = 0.5):
    # Get positions of cell bodies, which will be considered as vertices.
    props = measure.regionprops(cell_mask)
    centroid_list = []
    for region in props:
        centroid = (region.centroid)
        i = int(np.floor(centroid[0]))
        j = int(np.floor(centroid[1]))
        centroid_list.append((i,j))

    # Build matrix for cell bodies and neurites interpreted as a maze
    #     centroid of a cell >0 (source or target in maze)
    #     neurite and cell bodies = 0 (possible path in maze)
    #     background = -1 (wall in maze)
    large_maze = -1*(~((cell_mask + neurite_mask)>0)).astype(int)
    index = 0
    for centroid in centroid_list:
        index += 1
        large_maze[centroid] = index

    # Compress original maze to get smaller maze
    small_maze = -1*(matrix_binning( large_maze, compress_factor)<-threshold).astype(int)
    n = small_maze.shape[0]

    # Reposition the centroids into the small maze
    index = 0
    for centroid in centroid_list:
        index += 1
        i = centroid[0] // compress_factor
        j = centroid[1] // compress_factor
        if i<=n and j<=n:
            small_maze[i,j] = index


    return small_maze, large_maze

def matrix_binning( matrix, compress_factor):
    """
    Compress a 2d array by a factor 'compress_factor', a positive integer,
    along each dimension
    """
    true_dim = np.min(matrix.shape)
    for n in range(0,true_dim):
        cropping_dim = true_dim - n
        if (cropping_dim % compress_factor)==0:
            break

    cropped_matrix = matrix[0:cropping_dim, 0:cropping_dim]

    output_shape = (cropping_dim//compress_factor, cropping_dim//compress_factor)

    binning_shape = (output_shape[0], cropping_dim // output_shape[0],
                     output_shape[1], cropping_dim// output_shape[1])

    return cropped_matrix.reshape(binning_shape).mean(-1).mean(1)

@jit(nopython=True)
def propagate_sources(maze):
    """
    In a matrix, considered as a maze where -1 and 0 respectively stand for an element
    of a wall and a possible path, the function takes each positive integer, viewed as
    a source, and propagate its value to any adjacent element of value 0.
    """
    m, n = maze.shape
    # Elements where the sources can propagate their influence
    possible_moves = np.argwhere(maze==0)
    # Prepapring the maze to be updated with the propagation from the sources
    updated_maze = maze.copy()
    # Propagation
    for element in possible_moves:
        i = element[0]
        j = element[1]
        if i>0 and maze[i-1,j] > 0: # propagate index to the right
            updated_maze[i,j] = maze[i-1,j]
        if j>0 and maze[i,j-1] > 0: # propagate index down
            updated_maze[i,j] = maze[i,j-1]
        if i<m-1 and maze[i+1,j] > 0:# propagate index up
            updated_maze[i,j] = maze[i+1,j]
        if j<n-1 and maze[i,j+1] > 0: # propagate index to the left
            updated_maze[i,j] = maze[i,j+1]

    return updated_maze

@jit(nopython=True)
def find_index_intersections(maze):
    m, n = maze.shape
    possible_intersections = np.argwhere(maze>0)
    intersection_list = []
    for element in possible_intersections:
        i = element[0]
        j = element[1]
        if i<m-1 and maze[i,j] != maze[i+1,j] and maze[i+1,j]>0:
            index_pair = sorted((maze[i,j], maze[i+1,j]))
            k = index_pair[0]
            l = index_pair[1]
            intersection_list.append([(k,l), (i,j)])
        if j<n-1 and maze[i,j] != maze[i,j+1] and maze[i,j+1]>0:
            index_pair = sorted((maze[i,j], maze[i,j+1]))
            k = index_pair[0]
            l = index_pair[1]
            intersection = [(i,j),(i,j+1)]
            intersection_list.append([(k,l), (i,j)])

    return intersection_list

@jit(nopython=True)
def find_edge(maze):
    intersection_list = find_index_intersections(maze)
    edge_set = set()
    for intersection in intersection_list:
        edge_set.add(intersection[0])

    return edge_set



@jit(nopython=True)
def solve_maze(binary_matrix, source, target):
    """
    Takes a binary matrix, considered as a maze where the 1's belong to the walls,
    and finds the shortest path between two points, source and target.
    """
    maze = np.copy(binary_matrix)
    m, n = maze.shape
    max_itr = m + n
    # Starts from the source and propagate on all allowed coordinates while counting
    # the number of moves
    steps_matrix = np.zeros((m,n))
    steps_matrix[source] = 1
    for step in range(1,max_itr):
        steps_matrix = move_in_maze(steps_matrix, maze)
        if steps_matrix[target] >0 :
            break
    path_length = steps_matrix[target]
    path = []
    if path_length > 0:
        i, j = target
        path.append((i,j))
        step = path_length
        while step>1:
            if i > 0 and steps_matrix[i - 1,j] == step-1:
                i, j = i-1, j
            elif j > 0 and steps_matrix[i,j - 1] == step-1:
                i, j = i, j-1
            elif i < m - 1 and steps_matrix[i + 1,j] == step-1:
                i, j = i+1, j
            elif j < n - 1 and steps_matrix[i,j + 1] == step-1:
                i, j = i, j+1
            path.append((i, j))
            step -= 1

    path_matrix = np.zeros((m,n))
    for ij in path:
        path_matrix[ij] = 1

    return path, path_matrix

@jit(nopython=True)
def move_in_maze(steps_matrix, maze):
    """
    Auxiliary function for solve_maze which allows to move one step
    into the maze (binary matrix) given the previous steps (steps_matrix).
    """
    actual_step = steps_matrix.max()
    m, n = steps_matrix.shape
    for i in range(m):
        for j in range(n):
            if steps_matrix[i,j] == actual_step:
                if i>0 and steps_matrix[i-1,j] == 0 and maze[i-1,j] == 0:
                    steps_matrix[i-1,j] = actual_step + 1
                if j>0 and steps_matrix[i,j-1] == 0 and maze[i,j-1] == 0:
                      steps_matrix[i,j-1] = actual_step + 1
                if i<m-1 and steps_matrix[i+1,j] == 0 and maze[i+1,j] == 0:
                      steps_matrix[i+1,j] = actual_step + 1
                if j<n-1 and steps_matrix[i,j+1] == 0 and maze[i,j+1] == 0:
                       steps_matrix[i,j+1] = actual_step + 1

    return steps_matrix
