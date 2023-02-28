import numpy as np
import pandas as pd
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
file = '/Users/atanu/Desktop/S5120006_output_theta.csv'


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

df = getNodes(file)

pos = df['Center Node']

pos = remove_artificial_lines(pos)

def getPosPlot(pos):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pos.index // fps, pos, color='magenta', linewidth=2)
    ax.set(xlabel=r'$Time~[sec]$', ylabel=r'$Angular~position~[deg]$')
    ax.get_yaxis().set_label_coords(-0.1, 0.5)
    ax.get_xaxis().set_label_coords(0.5, -0.1)
    plt.tight_layout()
    plt.show()

getPosPlot(pos)
print('done!')
