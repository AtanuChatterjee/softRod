import numpy as np
import pandas as pd
import matplotlib as mpl
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, find_peaks

mpl.rcParams['font.size'] = 13
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.grid'] = True

fps = 25
start_frame = 25000
end_frame = 25010
file = 'Z:/Atanu/exp_2021_fluid_ants/soft_rods/white_rod/white_5cm_hinged/gif/S5120006_output_theta.csv'


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

def sort_angles(angles):
    # Sort the angles in a circular fashion
    angles = np.asarray(angles)
    sorted_indices = np.argsort(angles)
    sorted_angles = angles[sorted_indices]
    if sorted_angles[-1] - sorted_angles[0] >= 360.0:
        sorted_indices = np.roll(sorted_indices, -1)
        sorted_angles = angles[sorted_indices]
    return sorted_angles, sorted_indices

from scipy.interpolate import UnivariateSpline

def estimate_shape_and_curvature(csv_file):
    # Load CSV file into a pandas dataframe
    data = pd.read_csv(csv_file)
    # data = pd.read_csv(file)
    df = pd.DataFrame(data)
    df = df[(df['Frame_No'] >= start_frame) & (df['Frame_No'] <= end_frame)]
    # Get the number of nodes and frames from the dataframe
    num_nodes = df.shape[1] - 1
    num_frames = df.shape[0]

    # Initialize arrays to store node coordinates and curvature
    points = np.zeros((num_frames, num_nodes, 2))
    curvature = np.zeros((num_frames, num_nodes))

    # Loop over all frames and nodes to extract the coordinates
    for i in range(num_frames):
        for j, angle in enumerate(df.iloc[i, 1:]):
            # Calculate the x and y coordinates of the node using the angle
            theta = np.deg2rad(angle)
            if j == 0:
                points[i, j] = [0, 0]
            else:
                points[i, j] = points[i, j-1] + [np.cos(theta), np.sin(theta)]

    # Compute the spline functions for the x and y coordinates
    distance = np.linspace(0, 1, num_nodes)
    splines = [CubicSpline(distance, points[i]) for i in range(num_frames)]

    # Loop over all frames and nodes to calculate the curvature
    for i in range(num_frames):
        for j in range(num_nodes):
            if j >= 2:
                # Compute the tangent vector and its derivative
                tangent = splines[i](distance[j], 1)
                tangent /= np.linalg.norm(tangent)
                d_tangent = splines[i](distance[j], 2)

                # Compute the curvature using the formula: |dT/ds| / |R|
                curvature[i, j] = np.linalg.norm(np.cross(tangent, d_tangent)) / np.linalg.norm(d_tangent) ** 2

    # Plot the curve and its curvature at each node for the last frame
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('x')
    ax1.set_ylabel('y', color=color)
    print(points.shape)
    ax1.plot(points[-1, :, 0], points[-1, :, 1], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('curvature', color=color)
    print(curvature.shape)
    for i in range(num_frames):
        ax1.plot(points[i, :, 0], points[i, :, 1], color='red', alpha=0.5)
        ax2.plot(points[i, :, 0], curvature[i], color='blue', alpha=0.5)

    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()




estimate_shape_and_curvature(file)

# Compute the curvature numerically using the second derivative

# getPosPlot(pos)
#
# data = pd.read_csv(file)
# df = pd.DataFrame(data)
# df = df[(df['Frame_No'] >= start_frame) & (df['Frame_No'] <= end_frame)]
# theta_cols = df.filter(regex='^Node').columns.tolist()

# theta_cols = df.filter(regex='^theta').columns.tolist()

# Convert angles from degrees to radians
# df[theta_cols] = np.deg2rad(df[theta_cols])

# Calculate the arc length of the rod at each frame
# d = 1.0  # distance between adjacent nodes
# diff_theta = np.diff(pos[0], axis=1)
# arc_length = np.sum(np.sqrt(d**2 + np.sum(diff_theta**2, axis=1)))
#
# # Calculate the second derivative of theta with respect to arc length
# dtheta_ds = np.gradient(df[theta_cols].values, arc_length, axis=1)
# d2theta_ds2 = np.gradient(dtheta_ds, arc_length, axis=1)
#
# # Calculate the curvature
# curvature = np.abs(d2theta_ds2) / (1 + dtheta_ds**2)**(3/2)

# Plot the curvature as a function of frame number




# plt.ylabel('Curvature')
# plt.scatter(pos.index // fps, curvature)
plt.show()

plt.show()




print('done!')
