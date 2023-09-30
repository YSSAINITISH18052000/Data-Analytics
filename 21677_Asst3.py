import numpy as np
import scipy
import matplotlib.pyplot as plt
import os

if not os.path.exists('../data'):
    os.makedirs('../data')
if not os.path.exists('../plots'):
    os.makedirs('../plots')

def preprocess(path):
    # Using .readlines() to extract raw data from the .txt file line by line
    with open(path, "r") as data_file:
        raw_data = data_file.readlines()
    
    return raw_data

def convert_list_of_lists_to_float(list_of_lists):
  """Converts a list of lists where each inner list contains strings as elements into float.

  Args:
    list_of_lists: A list of lists where each inner list contains strings as elements.

  Returns:
    A list of lists where each inner list contains floats as elements.
  """

  new_list_of_lists = [[float(element) for element in inner_list] for inner_list in list_of_lists]
  return new_list_of_lists


def prune_data(raw_data):
    # Splitting the string text into columns of list 
    for i in range(len(raw_data)):
        raw_data[i] = raw_data[i].split()
    #print("\nThe number of data points in raw_data are",len(raw_data))
    #print("\nThe columns present in data are:",len(raw_data[0]))

    # Checking for incomplete data points
    Filtered_data = []
    for i in range(len(raw_data)):
        if(len(raw_data[i]) == len(raw_data[0])):
            Filtered_data.append(raw_data[i])
    #print("\nThe number of data points after filtering are:",len(Filtered_data))

    # Removing columns 'ProbeName', 'GeneSymbol', 'EntrezGeneID', 'GO' from the filtered_data 
    Processed_data = []
    for i in range(len(Filtered_data)):
        Processed_data.append(Filtered_data[i][1:49])
    
    return Processed_data

def F_statistic(Processed_data):
    # Constructing 'N' and 'D' matrices 
    N = np.zeros((48,4))
    D = np.zeros((48,4))

    a1 = [1,0,1,0]
    a2 = [1,0,0,1]
    a3 = [0,1,1,0]
    a4 = [0,1,0,1]

    for i in range(48):
        if(i<=11):
            N[i,:] = a2
            D[i,:] = [0,1,0,0]
        elif(i>11 and i<=23):
            N[i,:] = a1
            D[i,:] = [1,0,0,0]
        elif(i>23 and i<=35):
            N[i,:] = a4
            D[i,:] = [0,0,0,1]
        elif(i>35 and i<=47):
            N[i,:] = a3
            D[i,:] = [0,0,1,0]

    # Rank of N = No. of genders + No. of smoking statuses-1
    # Rank of D = No. of genders * No. of smoking statuses(beacuse no of linearly independent rows in D will be equal to as many as it) 
    rank_N = 3
    rank_D = 4

    ntn = np.matmul(np.transpose(N),N)
    ntn_dagger = np.linalg.pinv(ntn)
    ntn_extended_dagger = np.matmul(N, ntn_dagger)
    ntn_final = np.matmul(ntn_extended_dagger, np.transpose(N))
    mat1 = np.identity(48) - ntn_final

    dtd = np.matmul(np.transpose(D),D)
    dtd_dagger = np.linalg.pinv(dtd)
    dtd_extended_dagger = np.matmul(D, dtd_dagger)
    dtd_final = np.matmul(dtd_extended_dagger, np.transpose(D))
    mat2 = np.identity(48) - dtd_final

    # Using Linear Regression 
    Processed_data = convert_list_of_lists_to_float(Processed_data[1:])
    np.array(Processed_data)

    P = []
    for i in range(len(Processed_data)):
        
        num = 0
        num = np.matmul(np.transpose(Processed_data[i]), mat1)
        num = np.matmul(num, Processed_data[i])

        den = 0
        den = np.matmul(np.transpose(Processed_data[i]), mat2)
        den = np.matmul(den, Processed_data[i])

        result = (num/(den + 1e-7)) - 1
        F_statistic = 44 * result
        P.append(1 - scipy.stats.f.cdf(F_statistic,1,44))
    
    return P

def histogram(P, plot_path):
    plt.hist(P,bins = 20)
    plt.xlabel('p-values')
    plt.ylabel('Frequency')
    plt.title('Histogram of p-values')
    plt.savefig(plot_path)

def main():
    data_path = 'C:/Users/yvsdm/Data_Analytics/Assignment 3/data/Raw Data_GeneSpring.txt'
    plot_path = 'C:/Users/yvsdm/Data_Analytics/Assignment 3/plots/Histogram.png'
    raw_data = preprocess(data_path)
    Processed_data = prune_data(raw_data)
    P = F_statistic(Processed_data)
    histogram(P, plot_path)

    return 1

if __name__ == "__main__":
    main()