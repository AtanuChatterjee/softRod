import numpy as np
import pandas as pd
from scipy.fft import fft
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, find_peaks

mpl.rcParams['font.size'] = 13
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.grid'] = True

fps = 25
start_frame = 25000
end_frame = 40000
# file = '/Users/atanu/Desktop/S5120006_output_theta.csv'
file = '/Users/atanu/Documents/GitHub/softRod/softRodmodel/data/nv6/qData_l5_nv6.csv'


def getNodes(file):
    """
    Given a DataFrame `df` with frame no as the first column, and nodes 0 through `num_nodes-1`
    in the subsequent columns, this function extracts the head node, center node, and tail node
    as a function of frame no and plots them.
    """

    data = pd.read_csv(file)
    df = pd.DataFrame(data)
    df = df[(df['Frame_No'] >= start_frame) & (df['Frame_No'] <= end_frame)]

    num_nodes = len(df.columns) - 1

    # Determine the center node index
    if num_nodes % 2 == 0:
        center_node_idx = num_nodes // 2
    else:
        center_node_idx = num_nodes // 2 + 1

    # Extract the head, center, and tail node columns
    head_node_col = df.iloc[:, 2]
    center_node_col = df.iloc[:, center_node_idx]
    tail_node_col = df.iloc[:, num_nodes - 1]

    # Create a new DataFrame with the extracted columns
    new_df = pd.DataFrame({
        'Frame No': df.iloc[:, 0], 'Head Node': head_node_col, 'Center Node': center_node_col,
        'Tail Node': tail_node_col})

    new_df = new_df.reset_index(drop=True)

    return new_df

def remove_artificial_lines(pos):
    diffs = np.append(np.diff(pos), 0)
    discont_indices = np.abs(diffs) > 180
    pos[discont_indices] = np.nan
    return pos

# df = getNodes(file)
#
# pos = df['Center Node']
#
# pos = remove_artificial_lines(pos)

def getPosPlot(pos):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pos.index // fps, pos, color='magenta', linewidth=2)
    ax.set(xlabel=r'$Time~[sec]$', ylabel=r'$Angular~position~[deg]$')
    ax.get_yaxis().set_label_coords(-0.1, 0.5)
    ax.get_xaxis().set_label_coords(0.5, -0.1)
    plt.tight_layout()
    plt.show()

# getPosPlot(pos)


def plot_curvature_for_node(filename, node_idx):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(filename)

    # Extract the time and node coordinates from the DataFrame
    time = df.iloc[:, 0].values
    x = df.iloc[:, 1::2].values
    y = df.iloc[:, 2::2].values

    # Define the number of nodes and elements
    n_nodes = x.shape[1]

    # Loop over each time step
    for i in range(len(time)):
        # Define the boundary conditions
        x[i, 0] = 0.0
        y[i, 0] = 0.0

        # Define the deformation
        dx = x[i] - np.linspace(0, max(x[i]), n_nodes)
        dy = y[i] - y[i, 0]

        # Calculate the distance between adjacent nodes
        distance = np.sqrt((x[i, 1:] - x[i, :-1]) ** 2 + (y[i, 1:] - y[i, :-1]) ** 2)

        # Calculate the curvature at the desired node
        if node_idx < n_nodes:
            curvature = 2 * (dy[node_idx] / distance[node_idx - 1] ** 2 + dy[node_idx + 1] / distance[node_idx] ** 2)
        else:
            curvature = 0.0

        # Plot the curvature as a function of time
        plt.plot(time[i], curvature, 'o', color='b')

    # Add labels and titles to the plot
    plt.xlabel(r'$Time$')
    plt.ylabel(r'$Curvature$')
    plt.title(f'Curvature of node {node_idx}')
    plt.show()

def get_curvature_for_all_nodes(filename):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(filename)

    # Extract the time and node coordinates from the DataFrame
    time = df.iloc[:, 0].values
    x = df.iloc[:, 1::2].values
    y = df.iloc[:, 2::2].values

    # Define the number of nodes and elements
    n_nodes = x.shape[1]

    # Create an empty list to store the curvature values for each node
    curvature_data = []

    # Loop over each node
    for node_idx in range(n_nodes):
        # Create an empty DataFrame to store the curvature values for this node
        node_curvature_df = pd.DataFrame(columns=['time', 'node_idx', 'curvature'])

        # Loop over each time step
        for i in range(len(time)):
            # Define the boundary conditions
            x[i, 0] = 0.0
            y[i, 0] = 0.0

            # Define the deformation
            dx = x[i] - np.linspace(0, max(x[i]), n_nodes)
            dy = y[i] - y[i, 0]

            # Calculate the distance between adjacent nodes
            distance = np.sqrt((x[i, 1:] - x[i, :-1]) ** 2 + (y[i, 1:] - y[i, :-1]) ** 2)

            # Calculate the curvature for this node at this time step
            if node_idx < n_nodes - 1:
                curvature = 2 * (dy[node_idx] / distance[node_idx] ** 2 + dy[node_idx + 1] / distance[node_idx + 1] ** 2)
            else:
                curvature = 0.0

            # Append the curvature value to the node's curvature DataFrame
            node_curvature_df = node_curvature_df.append({'time': time[i], 'node_idx': node_idx, 'curvature': curvature}, ignore_index=True)

        # Append the node's curvature DataFrame to the list of curvature data for all nodes
        curvature_data.append(node_curvature_df)

    # Concatenate all the curvature DataFrames into one
    curvature_df = pd.concat(curvature_data)

    return curvature_df


def estimate_period(signal, sample_rate):
    """
    Estimates the period of a periodic signal using the Fourier transform.

    Parameters:
        signal (numpy.ndarray): A 1D numpy array representing the periodic signal.
        sample_rate (float): The sampling rate of the signal, in Hz.

    Returns:
        period (float): The estimated period of the signal, in seconds.
    """
    # Compute the Fourier transform of the signal
    spectrum = np.abs(fft(signal))

    # Find the index of the maximum frequency component in the spectrum
    max_index = np.argmax(spectrum)

    # Estimate the frequency of the signal
    frequency = max_index * sample_rate / len(signal)

    # Estimate the period of the signal
    period = 1 / frequency

    return period


# a = plot_curvature_for_node(file, 1)
a = get_curvature_for_all_nodes(file)
p = estimate_period(a['Node3'], 100)
print('done!')
