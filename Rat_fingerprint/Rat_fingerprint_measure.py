#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pdesrosiers, zyazdani-92
"""

import matplotlib.pyplot as plt  # Ensure matplotlib is imported
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from numba import jit
import numpy as np
import networkx as nx
from scipy.linalg import expm
from scipy.spatial import distance
from scipy.stats import skew, kurtosis, entropy, differential_entropy
from networkx.algorithms.community import greedy_modularity_communities, modularity
import matplotlib.pyplot as plt
import os
import hdbscan
import umap
import sklearn.cluster as cluster
import matplotlib as mpl
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
from sklearn import tree
from sklearn.cluster import DBSCAN
import pandas as pd
def fingerprint_measures(Adj):
    """
    Compute many statitics that characterize the topology of the (possibly weighted and directed) graph
    corresponding to the adjacency matrix given as argument

    Refs:

    1. Esfahlani et al. Local structure-function relationships in human brain networks across the lifespan.
        Nat Commun 13, 2053 (2022). https://doi.org/10.1038/s41467-022-29770-y

    2. Thibeault, V., Allard, A. and Desrosiers, P., 2022. The low-rank hypothesis of complex systems.
        arXiv preprint arXiv:2208.04848.

    3. Fornito, A., Zalesky, A. and Bullmore, E., 2016. Fundamentals of brain network analysis. Academic Press.

    4. brain-connectivity-toolbox.net

    5. Rubinov, M. and Sporns, O., 2010. Complex network measures of brain connectivity: uses and interpretations.
        Neuroimage, 52(3), pp.1059-1069.
    """
    #G = nx.DiGraph(Adj)   # DiGraph object, not all measures all well-defined for digraphs
    G = nx.Graph(Adj)
    #g = G.to_undirected() # Graph object


    # Find the connected components
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    counts = [len(c) for c in components]
    frac_largest_component = counts[0]/np.sum(counts)#fraction of nodes in the largest connected component
    ent_components = normalized_ent_counts(counts)#entropy for the distibution of nodes into the connected components

    # Analysis based on the largest connected component
    largest_component = components[0]
    G_c = G.subgraph(largest_component).copy() # digraph induced by largest connected  component

    measures = {'Density': nx.density(G),
                'Efficiency':nx.global_efficiency(G_c),
                '# of connected components':nx.number_connected_components(G),
                '# of communities':GN_communities(G),
                'Fraction of nodes in the largest connected component': frac_largest_component,
                'Connected components entropy': ent_components,
                'Fraction of nodes in the center': fraction_center(G_c),
                'Fraction of nodes in the periphery' : fraction_periphery(G_c),
                'Fraction of nodes in the barycenter': fraction_barycenter(G_c),
                'Transitivity': nx.transitivity(G),
                'Average clustering coefficient': nx.average_clustering(G),
                'Average square clustering': average_square_clustering(G),  
                'Modularity': modularity_analysis(G),
                'Assortativity coefficient analysis': assortativity_coefficient_analysis(G_c),
                'Shortest path length analysis': normalized_shortest_path_length_analysis(G_c),
                'Average eccentricity': eccentricity_analysis(G_c),
                'Average degree': Num_connections(G),
                'Average communicability': communicability_analysis(G_c),
                'Average node similarity': similarity_analysis(G_c),
                'Average node betweenness centrality':avg_node_betweenness_centrality(G),
                'Average edge betweenness centrality': avg_edge_betweenness_centrality(G)}

    return measures

measures_names = [
                'density',
                'efficiency',
                'Num_connected_component',
                'Num_clusters',
                'frac_largest_connected_component',
                'connected_components_ent',
                'frac_center',
                'frac_periphery',
                'frac_barycenter',
                'srank_ratio',
                'erank_ratio',
                'transitivity',
                'avg_clustering',
                'avg_square_clustering',
                'modularity',
                'assortativity_coefficient_analysis',
                'normalized_shortest_path_length_analysis',
                'markov_rate',
                'basic_stats_med',
                'basic_stats_avg',
                'basic_stats_std',
                'basic_stats_md',
                'basic_stats_skew',
                'basic_stats_kurt',
                'eccentricity_med',
                'eccentricity_avg',
                'eccentricity_std',
                'eccentricity_md',
                'eccentricity_skew',
                'eccentricity_kurt',
                'degree_analysis_med',
                'degree_analysis_avg',
                'degree_analysis_std',
                'degree_analysis_md',
                'degree_analysis_skew',
                'degree_analysis_kurt',
                'flow_graph_med',
                'flow_graph_avg',
                'flow_graph_std',
                'flow_graph_md',
                'flow_graph_skew',
                'flow_graph_kurt',
                'communicability_med',
                'communicability_avg',
                'communicability_std',
                'communicability_md',
                'communicability_skew',
                'communicability_kurt',
                'similarity_analysis_med',
                'similarity_analysis_avg',
                'similarity_analysis_std',
                'similarity_analysis_md',
                'similarity_analysis_skew',
                'similarity_analysis_kurt',
                'Avg_Node_betweenness_centrality',
                'Avg_edge_betweenness_centrality']


def normalize_adjacency(Adj, method = 'spectral'):
    """
    Normalize an adjacency matrix either by ensuring a spectral norm of 1
    or by making the largest matrix element equal to 1.
    """
    if method=='spectral':
        sigma_1 = np.linalg.svd(Adj, compute_uv=False)[0]
        normalized_Adj = Adj/sigma_1
    elif method=='max':
        normalized_Adj = Adj/Adj.max()

    return normalized_Adj

def dict_to_vec(dictionary):
    """
    Convert a dictionary, whose values are numbers or arrays of different size, into
    a row vector
    """
    vector = np.array([])
    for key in dictionary.keys():
        vector = np.hstack([vector,np.array(dictionary[key])])
    return vector


def basic_stats(data, with_entropy = False):
    """
    Compute 6 or 7 basic statistics that describe the distribution of the input data,
    collected in a n-dimenional array and converted into a one-dimensional array.
    The output contains: 
        - two measures of centrality, the median(med) and the mean or average (avg)
        - two measures of dispersion, the stadard deviation (std) and the mean absolute difference (md)
        - a measure of assymetry, the L-skewness denoted tau_3, which is normalized between -1 and 1
        - a measure of tailedness, the L-kurtosis denoted tau_4, which is normalized between -1 and 1
        - a measure of uncertainty, the normalized entropy, varying between 0 (sure result) and 1 (perferclty uniform ditribution)
        
    Arguments
    ---------
    data            numpy array, each element considered a a sample value from a random variable
    with_entropy    boolean, default False, if True the normalized entropy
    
    Returns
    -------
    statistics      dictionary with keys 'med', 'avd', 'std', 'md', 'tau_3', 'tau_4', and 'ent',
                    corresponding to the 7 measures explained above, and values being floats 
    """
    data_vector = (np.array(data)).flatten() # np.array to avoid problem with scipy matrix type

    # minval = data_vector.min()
    # maxval = data_vector.max()
    med = np.median(data_vector)
    avg = data_vector.mean()
#    std = data_vector.std()
    # skewness = skew(data_vector)
    # kurt = kurtosis(data_vector)
#     l_moments = L_moments(data_vector) # First four L-moments
#     md = 2*l_moments[1]                # Mean absolute difference
#     if md>0:
#         tau_3 = l_moments[2]/l_moments[1]  # L-skewness, a normalized skewness between -1 and 1
#         tau_4 = l_moments[3]/l_moments[1]  # L-kurtosis, a normalized kurtosis between -1 and 1 
#     else:
#         tau_3 = np.nan
#         tau_4 = np.nan
    
#     if with_entropy:
#         ent = normalized_ent(data_vector)    
#     else:
#         ent = np.nan

    statistics = {'med': med, 'avg': avg}#{'med': med, 'avg': avg, 'std': std, 'md': md, 'tau_3': tau_3, 'tau_4': tau_4, 'ent': ent}
    return statistics

@jit(nopython=True)
def L_moments(sample):
    """
    Estimate the first 4 L-moments of a sequence, considred as a sample 
    from a (one-dimensonal, continuous or discrete) random variable. The estimators
    are linear functions of the estimators for the order statistics.
    
    The 1st L-moment is equivalent to the mean, while the 2nd L-moment, also known as the L-scale,
    is equal to mean absolute difference divided by 2. 
    
    The r-th L-moment ratio for r>2 is defined as tau_r = (L-moment of order r)/(L-moment of order 2),  
    which is bounded between -1 and +1; tau_3 is the L-skewness and tau_4 is the L-kurtosis.
    
    The r-th L-moment, denoted lambda_r, is a LINEAR function of the order statistics. Contrary to the 
    standard moments, all r-th L-moments exist as long as the 1st one do. Moreover, any distribution 
    is uniquely characterized by its L-moments. 
    
    L-moments are used for fitting distributions such as the four-parameter kappa distribution [3,4,5]
    
    References:
    
    [1] Hosking, J.R., 1986. The theory of probability weighted moments (pp. 3-16). 
        New York, USA: IBM Research Division, TJ Watson Research Center.
        https://dominoweb.draco.res.ibm.com/reports/RC12210.pdf
        
    [2] Hosking, J.R., 1990. L‐moments: Analysis and estimation of distributions using linear combinations of order statistics. 
        Journal of the royal statistical society: series B (methodological), 52(1), pp.105-124.
        https://doi.org/10.1111/j.2517-6161.1990.tb01775.x
        
    [3] Hosking, J.R., 1994. The four-parameter kappa distribution. 
        IBM Journal of Research and Development, 38(3), pp.251-258.
    
    [4] Winchester, C., 2000. On estimation of the four-parameter kappa distribution. Master's thesis, Dalhousie University
        http://www.nlc-bnc.ca/obj/s4/f2/dsk2/ftp01/MQ57336.pdf
    
    [5] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kappa4.html
    """
    data = sample.flatten()
    size = len(data)
    ordered_data = np.sort(data)
    lstat1 = 0
    lstat2 = 0
    lstat3 = 0
    lstat4 = 0
    # Compute the L-statistics using the standard estimators
    for i in range(size):
        lstat1 += ordered_data[i] # 
        lstat2 += (numba_binom(i,1)-numba_binom(size-i-1,1))*ordered_data[i] # 
        lstat3 += (numba_binom(i,2) 
                   -2*numba_binom(i,1)*numba_binom(size-i-1,1) 
                   +numba_binom(size-i-1,2))*ordered_data[i]
        lstat4 += ( numba_binom(i,3) 
                   - 3*numba_binom(i,2)*numba_binom(size-i-1,1) 
                   + 3*numba_binom(i,1)*numba_binom(size-i-1,2)
                   - numba_binom(size-i-1,3) )*ordered_data[i]
    
    lstat1 /= numba_binom(size,1)
    lstat2 /= 2*numba_binom(size,2)
    lstat3 /= 3*numba_binom(size,3)
    lstat4 /= 4*numba_binom(size,4)
            
    return (lstat1, lstat2, lstat3, lstat4)

@jit(nopython=True)
def numba_binom(n,k):
    """ 
    Return the binomial coefficient of the positve integers n and k.
    Comptatible with other numba functions, contrary to scipy.stats.binom.
    Will provide the correct answer only if n and k are two positive integers such that n>=k
    """
    coeff = 1
    for i in range(k):
        coeff *= (n-i)/(k-i)
    return coeff

@jit(nopython=True)
def mean_abs_diff(sample):
    """ 
    Estimate the mean absolute diffrence (MD) of a sample from 
    a one-dimensional continuous random variable.
    """
    data = sample.flatten()
    size = len(data)
    MD = 0
    for i in range(size):
        for j in range(size):
            MD += np.abs(data[i]-data[j])
    MD /= size*(size-1)
    
    return MD

def normalized_ent(data_vector):
    """
    Compute the normalized entropy of a one-dimensional array of data, considred 
    as sample values from a continuous or discrete random variable.
    """
    values, counts = np.unique(data_vector, return_counts=True)
    # null entropy
    if len(values)==1:   
        ent = 0.0
    # continuous random variable
    elif len(values)>32: 
        gap = values.max()-values.min()
        if gap <= 1.0:
            ent = 0.0  # any continuous variable with support of length <1 has negative ent, which is neglected here
        else:
            ent = normalized_differential_ent(data_vector/divisor)
    # discrete random variable
    else:               
        ent = entropy(counts)/np.log(len(values))

    # bounding entropy between 0 and 1    
    if ent<0:
        ent=0.0
    elif ent>1:
        ent = 1.0

    return ent

def normalized_differential_ent(sample):
    """
    Estimate the normalized differential entropy of a sequence (sample from a one-dimensional
    continuous random variable) using the method proposed by Ebrahimi et al. [1] (1994), 
    which in general performs better than classical method of Vasicek [2](many estimators are reviewed in [3]).
    The normalization is computed using the uniform distribution on the range of values given by the sample, 
    thus assuming the underlying random variable has a bounded support. 
   
    References:
    
    [1] Ebrahimi, N., Pflughoeft, K. and Soofi, E.S., 1994. 
        Two measures of sample entropy. Statistics & Probability Letters, 20(3), pp.225-234
        https://doi.org/10.1016/0167-7152(94)90046-9Two measures of sample entropy. Statistics & Probability Letters, 20(3), pp.225-234
    
    [2] Vasicek, O., 1976. 
        A test for normality based on sample entropy. Journal of the Royal Statistical Society: Series B (Methodological), 38(1), pp.54-59.
        https://doi.org/10.1111/j.2517-6161.1976.tb01566.x
            
    [3] Noughabi, H. A. (2015). Entropy Estimation Using Numerical Methods.
        Annals of Data Science, 2(2), 231-241.
        https://link.springer.com/article/10.1007/s40745-015-0045-9
    """
    values = np.fallten(np.array(sample))
    gap = values.max()-values.min()
    if gap == 0:
        ent = 0.0
    else:
        ent = differential_entropy(values, method='ebrahimi')/np.log(gap) # denominator = maxent = ent of the uniform distribution
        if ent < 0.0:
            ent = 0.0
        elif ent > 1.0:
            end = 1.0
    
    return ent

def normalized_ent_counts(counts):
    """
    Compute the normalized entropy for a vector of counts (number of elements for each category)
    """
    if len(counts)==1:
        ent = 0
    else:
        ent = entropy(counts)/np.log(len(counts))

    if ent<0:
        ent=0
    elif ent>1:
        ent = 1

    return ent


def srank(G):
    """
    Compute the stable rank of the adjacency matrix related to the graph or digraph G
    Refs:
        V. Thibeault et al. The low-rank hypothesis, https://arxiv.org/abs/2208.04848, 2022.
        R. Vershynin, High-Dimensional Probability (Cambridge University Press, 2018).
    """
    A = nx.adjacency_matrix(G).todense()
    s = np.linalg.svd(A, full_matrices=True, compute_uv=False, hermitian=False)
    return np.sum(s**2)/s[0]**2

def normalized_srank(G):
    """
    Compute the normalized stable rank of the adjacency matrix related to the graph or digraph G
    If close to 0.0, then the graph is high dimensional; if close to 1.0, then low dimensional
    Refs:
        V. Thibeault et al. The low-rank hypothesis, https://arxiv.org/abs/2208.04848, 2022.
        R. Vershynin, High-Dimensional Probability (Cambridge University Press, 2018).
    """
    s = srank(G)
    N = G.number_of_nodes()
    return s/N

def erank(G, tolerance=1e-13):
    """
    Effective rank based on the definition using spectral entropy
    Refs:
         V. Thibeault et al. The low-rank hypothesis, https://arxiv.org/abs/2208.04848, 2022.
         https://ieeexplore.ieee.org/document/7098875 and doi:10.1186/1745-6150-2-2).
    """
    # We use the convention 0*log(0)=0 so we remove the zero singular values
    A = nx.adjacency_matrix(G).todense()
    singular_values = np.linalg.svd(A, full_matrices=True, compute_uv=False, hermitian=False)
    singular_values = singular_values[singular_values > tolerance]
    normalized_singular_values = singular_values / np.sum(singular_values)
    return np.exp(-np.sum(normalized_singular_values
                          * np.log(normalized_singular_values)))


def normalized_svd(G, tolerance=1e-13):
    """
    Compute singular values between 1 and 0 and their corresponding indices bewteen 0 and 1
    Refs:
         V. Thibeault et al. The low-rank hypothesis, https://arxiv.org/abs/2208.04848, 2022.
         https://ieeexplore.ieee.org/document/7098875 and doi:10.1186/1745-6150-2-2).
    """
    # We use the convention 0*log(0)=0 so we remove the zero singular values
    A = nx.adjacency_matrix(G).todense()
    N = A.shape[0]
    sv = np.linalg.svd(A, full_matrices=True, compute_uv=False, hermitian=False)
    normalized_sv = sv/sv[0]
    normalized_indices = np.array([i/N for i in range(N)])

    return  normalized_sv, normalized_indices

def normalized_erank(G):
    """
    Compute the normalized effective rank based on the definition using spectral entropy
    Refs:
         V. Thibeault et al. The low-rank hypothesis, https://arxiv.org/abs/2208.04848, 2022.
         https://ieeexplore.ieee.org/document/7098875 and doi:10.1186/1745-6150-2-2).
    """
    e = erank(G)
    N = G.number_of_nodes()
    return e/N

def fraction_center(G):
    """ Fraction of vertices contained in the center of a graph"""
    return len(nx.center(G))/G.number_of_nodes()

def fraction_periphery(G):
    """ Fraction of vertices that belong to the periphery of a graph"""
    return len(nx.periphery(G))/G.number_of_nodes()

def fraction_barycenter(G):
    """ Return the proportion of vertices in a connected graph that belong to the barycenter """
    return len(nx.barycenter(G))/G.number_of_nodes()

def eccentricity_analysis(g):
    """
    Compute the normalized eccentricities for all nodes of a connected graph
    and return basic statitistics describing the distribution of excenticities.
    """

    eccentricities = np.array(list(dict(nx.eccentricity(g)).values()))/(g.number_of_nodes()-1)
    
    
    return dict_to_vec(basic_stats(eccentricities))[:-1]


def degree_analysis(G):
    degrees = np.array(list(dict(G.degree()).values()))
    return dict_to_vec(basic_stats(degrees))[:-1]

def normalized_degree_analysis(G):
    degrees = np.array(list(dict(G.degree()).values()))/G.number_of_nodes()
    return dict_to_vec(basic_stats(degrees))[:-1]


def markov_analysis(g):
    A = nx.adjacency_matrix(g).todense()
    D =  np.diagflat(np.sum(A, axis = 1))
    L = np.eye(A.shape[0])- np.linalg.pinv(D)@A
    spectrum = np.sort(np.linalg.eigvals(L))
    convergence_rate = spectrum[1]
    return convergence_rate

def flow_graph_analysis(g):
    A = nx.adjacency_matrix(g).todense()
    D =  np.diagflat(np.sum(A, axis = 1))
    L = np.eye(A.shape[0])- np.linalg.pinv(D)@A
    transition_prob= expm(-L)
    return dict_to_vec(basic_stats(transition_prob))[:-1]


def communicability_analysis(g, return_matrix = False):
    """
    Compute basics statistics about the communicability of a NetworkX graph,
    concept defined in [1] for undirected and unweighted graphs and later 
    adapted in [2] for weighted networks by including normalization matrices.
    
    To avoid any convergence issue regarding the matrix exponential used to compute
    communicability, the graph's adjacency or weight matrix is made symmetric 
    and pseudo-inversion is used rather inversion. These operations don't 
    affect the result if the graph is undirected and all vertices have non-zero degree.
    
    References:
    [1] Estrada, E. and Hatano, N., 2008. Communicability in complex networks. Physical Review E, 77(3), p.036111.
        https://doi.org/10.1103/PhysRevE.77.036111
    [2] Crofts, J.J. and Higham, D.J., 2009. A weighted communicability measure applied to complex brain networks. 
        Journal of the Royal Society Interface, 6(33), pp.411-414.https://doi.org/10.1098/rsif.2008.0484
    """
    A = nx.adjacency_matrix(g).todense() # adjacency matrix
    A = 0.5*(A+A.T) # symmetrized matrix, ensuring proper diaginalization bu real orthogonal matrices
   
    D = np.diagflat(np.sum(A, axis = 1)) # degree matrix
    sqrtDminus1 = np.linalg.pinv(np.sqrt(D)) # (degree matrix)^(-1/2), normalization factor as in [2]
                                             # pseudo inversion used to avoid problem when some degrees are zero
    
    A_normalized = sqrtDminus1 @ A @sqrtDminus1 # normalized matrix as defined in [2]
    eigval, eigvec = np.linalg.eigh(A_normalized) # np.linalg.eigh works for hermitian matrices, ensuring real eigenvalues
    
    communicability_mat = np.array(eigvec @ np.diagflat(np.exp(eigval)) @ eigvec.T ) # scipy version, expm(A_normalized), 
                                                                                     # doesn't always converge to the right answer 

    communicability_stats = dict_to_vec(basic_stats(communicability_mat))[:-1]

    
    if return_matrix:
        return communicability_stats, communicability_mat, A
    else:
        return communicability_stats

def similarity_analysis(G):
    """
    Compute basics statistics describing the similarity of vertex neighbourhoods in a NetworkX 
    directed or undirected graph. The similariry of neighbourhoods is computed using the scalar product
    of row in the adjacency matrix.
    """
    A = np.array(nx.adjacency_matrix(G).todense())
    N = G.number_of_nodes()
    connected_nodes = tuple()
    for i in range(N):
        if np.linalg.norm(A[i,:])>0:
            connected_nodes += (i,)
    n = len(connected_nodes)
    A = A[np.ix_(connected_nodes, connected_nodes)]
    Sim = np.zeros((n,n))
    for i in range(n-1):
        for j in range(i+1,n):
            scalar = np.dot(A[i,:], A[j,:].T)
            Sim[i,j] = scalar/np.linalg.norm(A[i,:])/np.linalg.norm(A[j,:].T)
    Sim = Sim + Sim.T
    out = basic_stats(Sim)
    out = dict_to_vec(out)[:-1]
    return out


def average_square_clustering(G):
    """
    Compute the average over all square clustering coefficients of a NetworkX undirected graph
    using the function nx.square_clustering. The latter function doen't work 
    properly for directed graphs. Each square clustering coefficient is a number between 
    0 and 1 giving the fraction of possible squares that are observed in the graph.
    """
    dict_square_clustering= nx.square_clustering(G)
    square_clustering = np.array(list(dict_square_clustering.values()))
    avg = np.mean(square_clustering)
    if np.mean(square_clustering)>1:
        print('  Problem: average square clustering equal to '+ str(avg))
        avg = 1.0
    return avg

def modularity_analysis(G):
    partition = greedy_modularity_communities(G)
    mod = modularity(G,partition)
    return mod

def GN_communities(G):
    # Perform hierarchical clustering using Girvan-Newman algorithm
    clusters = nx.algorithms.community.girvan_newman(G)
    # Get the number of clusters
    num_communities = len(list(clusters))
    return num_communities



def assortativity_coefficient_analysis(G):
    assortativity = nx.degree_assortativity_coefficient(G)
    return assortativity


def normalized_shortest_path_length_analysis(G):
    num_nodes = G.number_of_nodes()  # Get the number of nodes in the subgraph
    shortest_path_length = nx.average_shortest_path_length(G)
    # Normalize the shortest path length
    normalized_shortest_path = shortest_path_length / (num_nodes - 1)
    return normalized_shortest_path

def avg_node_betweenness_centrality(G):
    node_betweenness_centrality = nx.betweenness_centrality(G, normalized=True)
    # Calculate the average node betweenness centrality
    average_node_betweenness_centrality = sum(node_betweenness_centrality.values())/len(node_betweenness_centrality)
    return average_node_betweenness_centrality


def avg_edge_betweenness_centrality(G):
    edge_betweenness_centrality = nx.edge_betweenness_centrality(G, normalized=True)
    # Calculate the average edge betweenness centrality
    average_edge_betweenness_centrality = sum(edge_betweenness_centrality.values())/len(edge_betweenness_centrality)
    return average_edge_betweenness_centrality

def Num_connections(G):
    degrees = [d for n, d in G.degree()]
    avg_degree = sum(degrees) / len(degrees)
    return  avg_degree

