import os
import pickle
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union



if not os.path.exists('../models'):
    os.makedirs('../models')
if not os.path.exists('../plots'):
    os.makedirs('../plots')


class DLModel:
    """
        Model Class to approximate the Z function as defined in the assignment.
    """

    def __init__(self):
        """Initialize the model."""
        self.Z0 = [None] * 10
        self.L = None
    
    def get_predictions(self, X, Z_0, w, L) -> np.ndarray:  ##
        """Get the predictions for the given X values.

        Args:
            X (np.array): Array of overs remaining values.
            Z_0 (float, optional): Z_0 as defined in the assignment.
                                   Defaults to None.
            w (int, optional): Wickets in hand.
                               Defaults to 10.
            L (float, optional): L as defined in the assignment.
                                 Defaults to None.
        
        Returns:
            np.array: Predicted score possible
        """
        
        runs_exp = run_func(X, L, Z_0)
        return runs_exp 
        

        pass

    def calculate_loss(self, Params, X, Y, w) -> float:   ##
        """ Calculate the loss for the given parameters and datapoints.
        Args:
            Params (list): List of parameters to be optimized.
            X (np.array): Array of overs remaining values.
            Y (np.array): Array of actual average score values.
            w (int, optional): Wickets in hand.
                               Defaults to 10.

        Returns:
            float: Mean Squared Error Loss for the model parameters 
                   over the given datapoints.
        """
        loss = np.sum((self.get_predictions(X, Params[w], w, Params[0]) - Y)**2)
        return loss
        pass
    
    def save(self, path):
        """Save the model to the given path.

        Args:
            path (str): Location to save the model.
        """
        with open(path, 'wb') as f:
            pickle.dump((self.L, self.Z0), f)
    
    def load(self, path):
        """Load the model from the given path.

        Args:
            path (str): Location to load the model.
        """
        with open(path, 'rb') as f:
            (self.L, self.Z0) = pickle.load(f)


def get_data(data_path) -> Union[pd.DataFrame, np.ndarray]:
    """
    Loads the data from the given path and returns a pandas dataframe.

    Args:
        path (str): Path to the data file.

    Returns:
        pd.DataFrame, np.ndarray: Data Structure containing the loaded data
    """
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(data_path)

    return df

    pass

# Create a function to modify the date format
def modify_date_format(date):
    if len(date) == 10:
        return date
    
    elif (len(date)==15):
        month = date[0:3]
        month_number = ""
        if(month == "Jan"):
            month_number = "01"
        elif(month == "Feb"):
            month_number = "02"
        elif(month == "Mar"):
            month_number = "03"
        elif(month == "Apr"):
            month_number = "04"
        elif(month == "May"):
            month_number = "05"
        elif(month == "Jun"):
            month_number = "06"
        elif(month == "Jul"):
            month_number = "07"
        elif(month == "Aug"):
            month_number = "08"
        elif(month == "Sep"):
            month_number = "09"
        elif(month == "Oct"):
            month_number = "10"
        elif(month == "Nov"):
            month_number = "11"
        elif(month == "Dec"):
            month_number = "12"
       
        return (date[4:6] + "-" + str(month_number) + "-" + date[11:])
    return date

def innings_filter(df):
    
    """
    Get a DataFrame where innings=1.

    Args:
        df (pd.DataFrame): The DataFrame to be filtered.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    # Select the rows where innings is equal to 1
    df = df[df["Innings"] == 1]
    
    return df

def over_filter(df):
    """
    Get a DataFrame where overs.

    Args:
        df (pd.DataFrame): The DataFrame to be filtered.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    df = innings_filter(df)
    
    # Select the rows where innings=1 & overs specified 
    df = df[df["Over"] == 1]
    df = df[df["Wickets.in.Hand"] < 10]

    column = ["Match","Over","Runs.Remaining","Innings.Total.Runs"]
    df_extracted = df[column]
    np_array = df_extracted.to_numpy()
    
    return np_array

def wickets_in_hand(wickets_in_hand, df):
    """
    Get a DataFrame where overs.

    Args:
        df (pd.DataFrame): The DataFrame to be filtered.
        wickets_in_hand: wickets remaining  in hand

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
        
    np_array1 = over_filter(df)
    df = innings_filter(df)
    
    # Select the rows where wickets_in_hand is as specified
    df = df[df["Wickets.in.Hand"] == wickets_in_hand]
    
    column = ["Match","Over","Runs.Remaining","Innings.Total.Runs"]
    df_extracted = df[column]
    np_array = df_extracted.to_numpy()
    
    if(wickets_in_hand == 10):
        size=np.shape(np_array)[0]
        
        for i in range(0, size):
            if(np_array[i][1] == 1):
                arr = np.zeros((1,4))
                arr[0][2] = np_array[i][3]
                arr[0][3] = np_array[i][3]
                np_array = np.insert(arr, 0, np_array, axis=0)
                
        np_array = np.insert(np_array1, 0, np_array, axis=0)
 
    for i in range(0, np.shape(np_array)[0]):
        np_array[i][1] = 50 - np_array[i][1]
    np_array = np_array[:, 1:3]
    
    return np_array

def last_over_filter(df):
    df = innings_filter(df)
    df = df[df["Over"] == 50]
    
    column = ["Over","Runs"]
    df_extracted = df[column]
    np_array = df_extracted.to_numpy()
    
    count = np.shape(np_array)[0]
    sum = np.sum(np_array, axis=0)[1]
    L = sum/count 
    
    return L

def preprocess_data(data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
    """Preprocesses the dataframe by
    (i)   removing the unnecessary columns,
    (ii)  loading date in proper format DD-MM-YYYY,
    (iii) removing the rows with missing values,
    (iv)  anything else you feel is required for training your model.

    Args:
        data (pd.DataFrame, nd.ndarray): Pandas dataframe containing the loaded data

    Returns:
        pd.DataFrame, np.ndarray: Datastructure containing the cleaned data.
    """
    # Apply the function to the dates column
    data["Date"] = data["Date"].apply(modify_date_format)
    column = ["Match", "Innings", "Over", "Innings.Total.Runs","Runs.Remaining", "Wickets.in.Hand"]
    data = data[column]
    
    return data

def run_func(u, L, z):
    return z*(1 - np.exp(-L*u/z))


def train_model(data: Union[pd.DataFrame, np.ndarray], model: DLModel) -> DLModel:
    """Trains the model

    Args:
        data (pd.DataFrame, np.ndarray): Datastructure containg the cleaned data
        model (DLModel): Model to be trained
    """
    params = [10]*11
    params[10] = 200 
    over_data = {}
    runs_data = {}
    for wickets in range(1, 11):
        data_extracted = wickets_in_hand(wickets,data)
        u_data = data_extracted[:, 0]
        run_data = data_extracted[:, 1]
        over_data[wickets] = u_data
        runs_data[wickets] = run_data
    
    popt = sp.optimize.minimize(loss, params, (over_data, runs_data))
    model.L = popt.x[0]
    model.Z0 = popt.x[1:]
 
    return model

def loss(params, over_data, runs_data):
    error = 0
    for wickets in range(1, 11):
        error += np.mean((run_func(over_data[wickets], params[0], params[wickets]) - runs_data[wickets])**2)  
    return error


def plot(model: DLModel, plot_path: str) -> None:
    """ Plots the model predictions against the number of overs
        remaining according to wickets in hand.

    Args:
        model (DLModel): Trained model
        plot_path (str): Path to save the plot
    """

    for k in range(1, 11):
        overs = np.linspace(0, 51, 51)
        y = run_func(overs, model.L, model.Z0[k-1])
        plt.plot(overs, y, label = 'wickets %d' %k)
    plt.xlabel('Overs remaining')
    plt.ylabel('Expected runs')
    plt.legend()
    plt.savefig(plot_path)

    pass


def print_model_params(model: DLModel) -> List[float]:
    '''
    Prints the 11 (Z_0(1), ..., Z_0(10), L) model parameters

    Args:
        model (DLModel): Trained model
    
    Returns:
        array: 11 model parameters (Z_0(1), ..., Z_0(10), L)

    '''
    retparams = [0]*11
    retparams[0] = model.L
    retparams[1:] = model.Z0
    params = {}
    params['L'] = model.L
    for wickets in range(1, 11):
        params[wickets] = model.Z0[wickets-1]
    print(params)
    return (retparams)


def calculate_loss(model: DLModel, data: Union[pd.DataFrame, np.ndarray]) -> float:
    '''
    Calculates the normalised squared error loss for the given model and data

    Args:
        model (DLModel): Trained model
        data (pd.DataFrame or np.ndarray): Data to calculate the loss on
    
    Returns:
        float: Normalised squared error loss for the given model and data
    '''
    loss = 0
    size = 0
    params = [0]*11
    params[0] = model.L
    params[1:10] = model.Z0
    for wickets in range(1, 11):
        data_extracted = wickets_in_hand(wickets, data)
        size =  size + np.shape(data_extracted)[0]
        u_data = data_extracted[:, 0]
        run_data = data_extracted[:, 1]
        loss = loss + model.calculate_loss(params, u_data, run_data, wickets)
    print("\nThe normalized Square error over all datapoints is ",loss/size)
    return loss/size


def main(args):
    """Main Function"""

    data = get_data(args['data_path'])  # Loading the data
    print("Data loaded.")
    
    # Preprocess the data
    data = preprocess_data(data)
    print("Data preprocessed.")
    
    model = DLModel()  # Initializing the model
    model = train_model(data, model)  # Training the model
    model.save(args['model_path'])  # Saving the model
    
    plot(model, args['plot_path'])  # Plotting the model
    
    # Printing the model parameters
    print("\nThe model parameters are: ")
    print_model_params(model)

    # Calculate the normalised squared error
    calculate_loss(model, data)


if __name__ == '__main__':
    args = {
        "data_path": "../data/04_cricket_1999to2011.csv",
        "model_path": "../models/model.pkl",  # ensure that the path exists
        "plot_path": "../plots/plot.png",  # ensure that the path exists
    }
    main(args)
