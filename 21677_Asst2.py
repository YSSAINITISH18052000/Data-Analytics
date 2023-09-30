import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from copy import deepcopy
import seaborn as sb
import networkx as nx
import os
from copy import deepcopy
import pandas as pd
import time

if not os.path.exists('../plots'):
    os.makedirs('../plots')

def import_facebook_data(facebook_path):
    with open(facebook_path, "r") as f:
        data = np.loadtxt(f, dtype=np.int32)
        
    # Close the text file
    f.close()
    
    # Check for duplicate edges.
    data1 = np.sort(data)
    print("\nNumber of edges before removing duplicates are",len(data1))
    data2 = np.unique(data1, axis=0)
    print("\nNumber of edges after removing duplicates are",len(data2))

    return data2

def import_bitcoin_data(path):
    excel_file_path = path
    df = pd.read_csv(excel_file_path, header = None)
    column_names = ['Column1', 'Column2', 'Column3', 'Column4']
    df.columns = column_names
    selected_columns = df[['Column1','Column2']]
    selected_columns = np.array(selected_columns)
    selected_columns = np.sort(selected_columns)
    selected_columns = np.unique(selected_columns,axis=0)
    
    nodes = np.unique(selected_columns)
    # Create a mapping from unique elements to indices using a dictionary
    element_to_index = {element: index for index, element in enumerate(nodes)}
    for i in range(len(selected_columns)):
        a = selected_columns[i,0]
        selected_columns[i,0] = element_to_index[a]
        a = selected_columns[i,1]
        selected_columns[i,1] = element_to_index[a]
    
    return selected_columns


def spectralDecomp_OneIter(nodes_connectivity_list):
    nodes = np.unique(nodes_connectivity_list)
    total_nodes = len(nodes)
    # Create a mapping from unique elements to indices using a dictionary
    element_to_index = {element: index for index, element in enumerate(nodes)}
    adj_mat = np.zeros((total_nodes, total_nodes))
    degree_mat = np.zeros((total_nodes, total_nodes))
    
    #Computing Adjacency Matrix 
    edge = [0,0]
    for i in range(len(nodes_connectivity_list)):
        edge[0] = nodes_connectivity_list[i][0]
        edge[1] = nodes_connectivity_list[i][1]
        adj_mat[element_to_index[edge[0]]][element_to_index[edge[1]]] = 1
        adj_mat[element_to_index[edge[1]]][element_to_index[edge[0]]] = 1
    
    #Computing Degree Matrix
    degree_mat = np.diag(np.sum(adj_mat, axis = 1))  
    #print("\nThe adjacency matrix is\n",adj_mat)
    #print("\nThe degree matrix is\n",degree_mat)    
    

    #Computing Laplacian Matrix
    Laplacian = degree_mat - adj_mat
    #print("\nThe Laplacian Matrix is\n",Laplacian)

    #Computing the Fiedler vector
    eigen_values, eigen_vectors = eigh(Laplacian,degree_mat)
    eig_sorted = np.argsort(eigen_values)
    sorted_vectors = eigen_vectors[:,eig_sorted]
    Fiedler_Vector = sorted_vectors[:,1]
    #print("\nThe second smallest eigen value is corresponding eigen vector\n",eig)
    #print("\nCorresponding eigen vector is \n", Fiedler_Vector)
    
    #Computing the Graph Partition
    Graph_partition = np.zeros((total_nodes,2), dtype='int')
    
    Graph_partition[:,0] = nodes
    Graph_partition[:,1] = np.sign(Fiedler_Vector).reshape(-1)

    
    min1 = np.max(Graph_partition[:,0])+1
    min2 = np.max(Graph_partition[:,0])+1
    for i in range(len(Graph_partition)):
        if(Graph_partition[i][1] == -1 or Graph_partition[i][1] == 0):
            if(min1 > Graph_partition[i][0]):
                min1 = Graph_partition[i][0]
        elif(Graph_partition[i][1] == 1):
            if(min2  > Graph_partition[i][0]):
                min2 = Graph_partition[i][0]
    
    for i in range(len(Graph_partition)):
        if(Graph_partition[i][1] == -1 or Graph_partition[i][1] == 0):
            Graph_partition[i][1] = min1
        elif(Graph_partition[i][1] == 1):
            Graph_partition[i][1] = min2

    #Printing the Graph Partition
    #print("\nThe Graph Partition\n",Graph_partition)
    
    return Fiedler_Vector, adj_mat, Graph_partition

def my_spectral(nodes_connectivity_list, depth):
    if(depth>0):
        depth = depth - 1
        NA1, NA2, graph_partition = spectralDecomp_OneIter(nodes_connectivity_list)
        
        #Creating 2 new lists to store edges within each community 
        CID, CID_count = np.unique(graph_partition[:,1], return_counts=True)
        #print("\nCommunity IDs are",CID,"\nNumber of nodes in communites are", CID_count)
        edge_list1 = []
        edge_list2 = []
        community_1_nodes = graph_partition[graph_partition[:, 1] == CID[0]][:, 0]
        community_2_nodes = graph_partition[graph_partition[:, 1] == CID[1]][:, 0]

        # Create matrices for Community 1 and Community 2 edges
        community_1_edges = nodes_connectivity_list[np.isin(nodes_connectivity_list[:, 0], community_1_nodes) & np.isin(nodes_connectivity_list[:, 1], community_1_nodes)]
        community_2_edges = nodes_connectivity_list[np.isin(nodes_connectivity_list[:, 0], community_2_nodes) & np.isin(nodes_connectivity_list[:, 1], community_2_nodes)]
        
        edge_list1 = community_1_edges
        edge_list2 = community_2_edges
        #print("\nThe edge_list_1 is\n",edge_list1)
        #print("\nThe edge_list_2 is\n",edge_list2)
        #print("\nThe number of nodes at depth",depth+1,"in the community_1 are",len(np.unique(edge_list1)),"in the community_2 are",len(np.unique(edge_list2)))
        #print("\nAt depth = ",depth+1," Number of egdes in community_1 are:",len(edge_list1)," & Number of edges in community_2:",len(edge_list2))
        

        edges1 = np.array(edge_list1)
        edges2 = np.array(edge_list2)

        graph_partitions1 = my_spectral(edges1,depth)
        graph_partitions2 = my_spectral(edges2,depth)
 
        if(depth >0):
            graph_partition = np.row_stack((graph_partitions1,graph_partitions2))
       
        return graph_partition
    else:
        return 
    

def spectralDecomposition(nodes_connectivity_list):
    depth = 3
    full_graph_partition = my_spectral(nodes_connectivity_list, depth)
    print("\nThe graph partitioning after successive partitioning is\n",full_graph_partition)
    print(np.unique(full_graph_partition[:,1],return_counts=True))
    
    return full_graph_partition


def createSortedAdjMat(graph_partition_fb, nodes_connectivity_list_fb):
    indices = np.argsort(graph_partition_fb[:,1], axis=0)
    argnodes = graph_partition_fb[:,0][indices]
    nodes = np.unique(nodes_connectivity_list_fb)
    total_nodes = len(nodes)
    print(total_nodes)
    adj_mat = np.zeros((total_nodes, total_nodes))
        
    edge = [0,0]
    for i in range(len(nodes_connectivity_list_fb)):
        edge[0] = nodes_connectivity_list_fb[i][0]
        edge[1] = nodes_connectivity_list_fb[i][1]
        adj_mat[edge[0]][edge[1]] = 1
        adj_mat[edge[1]][edge[0]] = 1
    adj_mat = np.take(adj_mat,indices=argnodes,axis=1)
    adj_mat = np.take(adj_mat,indices=argnodes,axis=0)
    plt.figure()
    sb.heatmap(adj_mat)
    plt.savefig("Q3_Adjacency.png")

    #Graph Visualization
    G = nx.Graph()

    for node, community in graph_partition_fb:
        G.add_node(node, community = community)

    for i in range(len(graph_partition_fb)):
        for j in range(i+1, len(graph_partition_fb)):
            node1, community1 = graph_partition_fb[i]
            node2, community2 = graph_partition_fb[j]
            if(adj_mat[i][j] == 1):
                G.add_edge(node1, node2)

    # Extract community labels for coloring nodes
    community_labels = [G.nodes[node]['community'] for node in G.nodes]

    # Draw the graph with community-based node coloring
    pos = nx.spring_layout(G)  # You can use different layout algorithms
    plt.figure()
    nx.draw(G, pos, node_color=community_labels, cmap=plt.cm.Set1, with_labels=False)
    plt.savefig("Q3_Graph_Partition.png")

    return adj_mat


def compute_community_parameters(graph, communities):
    A_norm = nx.to_numpy_array(graph, dtype=int) / (2 * graph.number_of_edges())
    degree = np.sum(A_norm, axis=1)
    return A_norm, degree

def compute_merge_score(node, C, communities, A_norm, degree):
    communityC_nodes = np.where(communities == C)[0]
    sigma_total = sum(degree[node] for node in communityC_nodes)
    k_i_in = 2 * np.sum(A_norm[node, communityC_nodes])
    k_i = degree[node]
    Q_merge = k_i_in - 2 * sigma_total * k_i
    return Q_merge

def compute_demerge_score(node, communities, A_norm, degree):
    C = communities[node]
    communityC_nodes = np.where(communities == C)[0]
    sigma_total = sum(degree[node] for node in communityC_nodes)
    k_i_out = 2 * np.sum(A_norm[node, communityC_nodes])
    k_i = degree[node]
    Q_demerge = 2 * k_i * sigma_total - 2 * k_i ** 2 - k_i_out
    return Q_demerge

def compute_modularity_score(communities, A_norm, degree):
    community_id = np.unique(communities)
    Q = 0
    
    for C in community_id:
        communityC_nodes = np.where(communities == C)[0]
        sigma_total = sum(degree[node] for node in communityC_nodes)
        sigma_in = np.sum(A_norm[np.ix_(communityC_nodes, communityC_nodes)])
        Q += sigma_in - sigma_total ** 2
    
    return Q

def louvain_algorithm(graph):
    n = graph.number_of_nodes()
    communities = np.arange(n)
    graph_partition = np.zeros((n,2))
    graph_partition[:,0] = communities
    A_norm, degree = compute_community_parameters(graph, communities)
    arr = np.arange(n)
    iteration = 0
    
    while True:
        iteration += 1
        count = 0
        #np.random.shuffle(arr)
        
        for node in arr:
            neighbour_communities = np.unique(communities[np.where(A_norm[node] != 0)[0]])
            Q_demerge = compute_demerge_score(node, communities, A_norm, degree)
            Q_max = 0
            best_community = communities[node]
            
            for C in neighbour_communities:
                if C == communities[node]:
                    continue
                
                Q_merge = compute_merge_score(node, C, communities, A_norm, degree)
                delta_Q = Q_demerge + Q_merge
                
                if delta_Q > Q_max:
                    Q_max = delta_Q
                    best_community = C
            
            if Q_max > 0 and best_community != communities[node]:
                communities[node] = best_community
                count += 1
        
        print(f"Iteration: {iteration}, Community Count: {len(np.unique(communities))}, Modularity: {compute_modularity_score(communities, A_norm, degree)}")
        
        if count == 0:
            break
    graph_partition[:,1] = communities
    return graph_partition


def louvain_one_iter(nodes_connectivity_list):
    graph = nx.Graph()
    graph.add_edges_from(nodes_connectivity_list)
    graph_partition = louvain_algorithm(graph)

    return graph_partition

def main():   

    ############ Answer qn 1-4 for facebook data #################################################
    # Import facebook_combined.txt
    # nodes_connectivity_list is a nx2 numpy array, where every row 
    # is a edge connecting i<->j (entry in the first column is node i, 
    # entry in the second column is node j)
    # Each row represents a unique edge. Hence, any repetitions in data must be cleaned away.
    nodes_connectivity_list_fb = import_facebook_data("facebook_combined.txt")

    # This is for question no. 1
    # fielder_vec    : n-length numpy array. (n being number of nodes in the network)
    # adj_mat        : nxn adjacency matrix of the graph
    # graph_partition: graph_partitition is a nx2 numpy array where the first column consists of all
    #                  nodes in the network and the second column lists their community id (starting from 0)
    #                  Follow the convention that the community id is equal to the lowest nodeID in that community.
    
    fielder_vec_fb, adj_mat_fb, graph_partition_fb = spectralDecomp_OneIter(nodes_connectivity_list_fb)
    
    
    #Plotting sorted Fiedler vector
    sorted_fiedler_vector = np.sort(fielder_vec_fb, axis=0)
    x = np.unique(nodes_connectivity_list_fb)
    y = sorted_fiedler_vector
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel('NODE')
    plt.ylabel('Fiedler value')
    plt.savefig('../plots/Q1_Sorted_Fiedler_Vector_fb.png')

    #Plotting associated Adjacency Matrix
    args = np.argsort(fielder_vec_fb, axis=0).reshape(-1)
    Adj = deepcopy(adj_mat_fb)
    Adj = np.take(Adj,indices=args,axis=1)
    Adj = np.take(Adj,indices=args,axis=0)
    plt.figure()
    sb.heatmap(Adj)
    plt.savefig('../plots/Q1_Adjcency_Matrix_fb.png')

    #Graph Visualization
    G = nx.Graph()

    for node, community in graph_partition_fb:
        G.add_node(node, community = community)

    for i in range(len(graph_partition_fb)):
        for j in range(i+1, len(graph_partition_fb)):
            node1, community1 = graph_partition_fb[i]
            node2, community2 = graph_partition_fb[j]
            if(adj_mat_fb[i][j] == 1):
                G.add_edge(node1, node2)

    # Extract community labels for coloring nodes
    community_labels = [G.nodes[node]['community'] for node in G.nodes]

    # Draw the graph with community-based node coloring
    pos = nx.spring_layout(G)  # You can use different layout algorithms
    plt.figure()
    nx.draw(G, pos, node_color=community_labels, cmap=plt.cm.Set1, with_labels=False)
    plt.savefig('../plots/Q1_Graph_partition_fb.png')



    # This is for question no. 2. Use the function 
    # written for question no.1 iteratetively within this function.
    # graph_partition is a nx2 numpy array, as before. It now contains all the community id's that you have
    # identified as part of question 2. The naming convention for the community id is as before.
    start_time = time.time()
    graph_partition_fb = spectralDecomposition(nodes_connectivity_list_fb)
    end_time = time.time()
    print("\nElapsed time for FB SPECTRAL is",end_time - start_time)

    # This is for question no. 3
    # Create the sorted adjacency matrix of the entire graph. You will need the identified communities from
    # question 3 (in the form of the nx2 numpy array graph_partition) and the nodes_connectivity_list. The
    # adjacency matrix is to be sorted in an increasing order of communitites.
    clustered_adj_mat_fb = createSortedAdjMat(graph_partition_fb, nodes_connectivity_list_fb)

    indices = np.argsort(graph_partition_fb[:,1], axis=0)
    argnodes = graph_partition_fb[:,0][indices]
    nodes = np.unique(nodes_connectivity_list_fb)
    total_nodes = len(nodes)
    print(total_nodes)
    adj_mat = np.zeros((total_nodes, total_nodes))
        
    edge = [0,0]
    for i in range(len(nodes_connectivity_list_fb)):
        edge[0] = nodes_connectivity_list_fb[i][0]
        edge[1] = nodes_connectivity_list_fb[i][1]
        adj_mat[edge[0]][edge[1]] = 1
        adj_mat[edge[1]][edge[0]] = 1
    adj_mat = np.take(adj_mat,indices=argnodes,axis=1)
    adj_mat = np.take(adj_mat,indices=argnodes,axis=0)
    plt.figure()
    sb.heatmap(adj_mat)
    plt.savefig("../plots/Q3_Adjacency_fb.png")

    #Graph Visualization
    G = nx.Graph()

    for node, community in graph_partition_fb:
        G.add_node(node, community = community)

    for i in range(len(graph_partition_fb)):
        for j in range(i+1, len(graph_partition_fb)):
            node1, community1 = graph_partition_fb[i]
            node2, community2 = graph_partition_fb[j]
            if(adj_mat[i][j] == 1):
                G.add_edge(node1, node2)

    # Extract community labels for coloring nodes
    community_labels = [G.nodes[node]['community'] for node in G.nodes]

    # Draw the graph with community-based node coloring
    pos = nx.spring_layout(G)  # You can use different layout algorithms
    plt.figure()
    nx.draw(G, pos, node_color=community_labels, cmap=plt.cm.Set1, with_labels=False)
    plt.savefig("../plots/Q3_Graph_Partition_fb.png")


    # This is for question no. 4
    # run one iteration of louvain algorithm and return the resulting graph_partition. The description of
    # graph_partition vector is as before.
    start_time = time.time()
    graph_partition_louvain_fb = louvain_one_iter(nodes_connectivity_list_fb)
    end_time = time.time()
    print("\nElapsed time for FB LOUVAIN is", end_time-start_time)
    indices = np.argsort(graph_partition_louvain_fb[:,1], axis=0)
    argnodes = graph_partition_louvain_fb[:,0][indices]
    nodes = np.unique(nodes_connectivity_list_fb)
    total_nodes = len(nodes)
    print(total_nodes)
    adj_mat_louvain_fb = np.zeros((total_nodes, total_nodes))
    adj_mat_louvain_fb =adj_mat_louvain_fb.astype(np.int64)
    argnodes = argnodes.astype(np.int64)
        
    edge = [0,0]
    for i in range(len(nodes_connectivity_list_fb)):
        edge[0] = nodes_connectivity_list_fb[i][0]
        edge[1] = nodes_connectivity_list_fb[i][1]
        adj_mat_louvain_fb[edge[0]][edge[1]] = 1
        adj_mat_louvain_fb[edge[1]][edge[0]] = 1
    adj_mat_louvain_fb = np.take(adj_mat_louvain_fb,indices=argnodes,axis=1)
    adj_mat_louvain_fb = np.take(adj_mat_louvain_fb,indices=argnodes,axis=0)
    plt.figure()
    sb.heatmap(adj_mat_louvain_fb)
    plt.savefig("../plots/Q4_Adjacency_fb.png")

    #Graph Visualization
    G = nx.Graph()

    for node, community in graph_partition_louvain_fb:
        G.add_node(node, community = community)

    for i in range(len(graph_partition_fb)):
        for j in range(i+1, len(graph_partition_louvain_fb)):
            node1, community1 = graph_partition_louvain_fb[i]
            node2, community2 = graph_partition_louvain_fb[j]
            if(adj_mat_louvain_fb[i][j] == 1):
                G.add_edge(node1, node2)

    # Extract community labels for coloring nodes
    community_labels = [G.nodes[node]['community'] for node in G.nodes]

    # Draw the graph with community-based node coloring
    pos = nx.spring_layout(G)  # You can use different layout algorithms
    plt.figure()
    nx.draw(G, pos, node_color=community_labels, cmap=plt.cm.Set1, with_labels=False)
    plt.savefig("../plots/Q4_Graph_Partition_fb.png")

    


    ############ Answer qn 1-4 for bitcoin data #################################################
    # Import soc-sign-bitcoinotc.csv
    nodes_connectivity_list_btc = import_bitcoin_data("soc-sign-bitcoinotc.csv")

    # Question 1
    
    fielder_vec_btc, adj_mat_btc, graph_partition_btc = spectralDecomp_OneIter(nodes_connectivity_list_btc)
     #Plotting sorted Fiedler vector
    sorted_fiedler_vector = np.sort(fielder_vec_btc, axis=0)
    x = np.unique(nodes_connectivity_list_btc)
    y = sorted_fiedler_vector
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel('NODE')
    plt.ylabel('Fiedler value')
    plt.savefig('../plots/Q1_Sorted_Fiedler_Vector_btc.png')

    #Plotting associated Adjacency Matrix
    args = np.argsort(fielder_vec_btc, axis=0).reshape(-1)
    Adj = deepcopy(adj_mat_btc)
    Adj = np.take(Adj,indices=args,axis=1)
    Adj = np.take(Adj,indices=args,axis=0)
    plt.figure()
    sb.heatmap(Adj)
    plt.savefig('../plots/Q1_Adjcency_Matrix_btc.png')

    #Graph Visualization
    G = nx.Graph()

    for node, community in graph_partition_btc:
        G.add_node(node, community = community)

    for i in range(len(graph_partition_btc)):
        for j in range(i+1, len(graph_partition_btc)):
            node1, community1 = graph_partition_btc[i]
            node2, community2 = graph_partition_btc[j]
            if(adj_mat_btc[i][j] == 1):
                G.add_edge(node1, node2)

    # Extract community labels for coloring nodes
    community_labels = [G.nodes[node]['community'] for node in G.nodes]

    # Draw the graph with community-based node coloring
    pos = nx.spring_layout(G)  # You can use different layout algorithms
    plt.figure()
    nx.draw(G, pos, node_color=community_labels, cmap=plt.cm.Set1, with_labels=False)
    plt.savefig('../plots/Q1_Graph_partition_btc.png')
   
    # Question 2
    start_time = time.time()
    graph_partition_btc = spectralDecomposition(nodes_connectivity_list_btc)
    end_time = time.time()
    print("\nElapsed time for BTC SPECTRAL is",end_time-start_time)

    # Question 3
    clustered_adj_mat_btc = createSortedAdjMat(graph_partition_btc, nodes_connectivity_list_btc)

    indices = np.argsort(graph_partition_btc[:,1], axis=0)
    argnodes = graph_partition_btc[:,0][indices]
    nodes = np.unique(nodes_connectivity_list_btc)
    total_nodes = len(nodes)
    print(total_nodes)
    adj_mat = np.zeros((total_nodes, total_nodes))
        
    edge = [0,0]
    for i in range(len(nodes_connectivity_list_btc)):
        edge[0] = nodes_connectivity_list_btc[i][0]
        edge[1] = nodes_connectivity_list_btc[i][1]
        adj_mat[edge[0]][edge[1]] = 1
        adj_mat[edge[1]][edge[0]] = 1
    adj_mat = np.take(adj_mat,indices=argnodes,axis=1)
    adj_mat = np.take(adj_mat,indices=argnodes,axis=0)
    plt.figure()
    sb.heatmap(adj_mat)
    plt.savefig("../plots/Q3_Adjacency_btc.png")

    #Graph Visualization
    G = nx.Graph()

    for node, community in graph_partition_btc:
        G.add_node(node, community = community)

    for i in range(len(graph_partition_btc)):
        for j in range(i+1, len(graph_partition_btc)):
            node1, community1 = graph_partition_btc[i]
            node2, community2 = graph_partition_btc[j]
            if(adj_mat[i][j] == 1):
                G.add_edge(node1, node2)

    # Extract community labels for coloring nodes
    community_labels = [G.nodes[node]['community'] for node in G.nodes]

    # Draw the graph with community-based node coloring
    pos = nx.spring_layout(G)  # You can use different layout algorithms
    plt.figure()
    nx.draw(G, pos, node_color=community_labels, cmap=plt.cm.Set1, with_labels=False)
    plt.savefig("../plots/Q3_Graph_Partition_btc.png")


    # Question 4
    start_time = time.time()
    graph_partition_louvain_btc = louvain_one_iter(nodes_connectivity_list_btc)
    end_time = time.time()
    print("\nElapsed time for BTC LOUVAIN is",end_time-start_time)
    indices = np.argsort(graph_partition_louvain_btc[:,1], axis=0)
    argnodes = graph_partition_louvain_btc[:,0][indices]
    nodes = np.unique(nodes_connectivity_list_btc)
    total_nodes = len(nodes)
    print(total_nodes)
    adj_mat_louvain_btc = np.zeros((total_nodes, total_nodes))
    adj_mat_louvain_btc =adj_mat_louvain_btc.astype(np.int64)
    argnodes = argnodes.astype(np.int64)
        
    edge = [0,0]
    for i in range(len(nodes_connectivity_list_btc)):
        edge[0] = nodes_connectivity_list_btc[i][0]
        edge[1] = nodes_connectivity_list_btc[i][1]
        adj_mat_louvain_btc[edge[0]][edge[1]] = 1
        adj_mat_louvain_btc[edge[1]][edge[0]] = 1
    adj_mat_louvain_btc = np.take(adj_mat_louvain_btc,indices=argnodes,axis=1)
    adj_mat_louvain_btc = np.take(adj_mat_louvain_btc,indices=argnodes,axis=0)
    plt.figure()
    sb.heatmap(adj_mat_louvain_btc)
    plt.savefig("../plots/Q4_Adjacency_btc.png")

    #Graph Visualization
    G = nx.Graph()#

    for node, community in graph_partition_louvain_btc:
        G.add_node(node, community = community)

    for i in range(len(graph_partition_btc)):
        for j in range(i+1, len(graph_partition_louvain_btc)):
            node1, community1 = graph_partition_louvain_btc[i]
            node2, community2 = graph_partition_louvain_btc[j]
            if(adj_mat_louvain_btc[i][j] == 1):
                G.add_edge(node1, node2)

    # Extract community labels for coloring nodes
    community_labels = [G.nodes[node]['community'] for node in G.nodes]

    # Draw the graph with community-based node coloring
    pos = nx.spring_layout(G)  # You can use different layout algorithms
    plt.figure()
    nx.draw(G, pos, node_color=community_labels, cmap=plt.cm.Set1, with_labels=False)
    plt.savefig("../plots/Q4_Graph_Partition_btc.png")


if __name__ == "__main__":
    main()