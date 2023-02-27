import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.size'] = 13
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.grid'] = True

start_frame = 25000
end_frame = 40000
file = 'Z:/Atanu/exp_2021_fluid_ants/soft_rods/white_rod/white_5cm_hinged/gif/S5120006_output_theta.csv'

data = pd.read_csv(file)
df = pd.DataFrame(data)
df = df[(df['Frame_No'] >= start_frame) & (df['Frame_No'] <= end_frame)]
fig, ax = plt.subplots(figsize=(15, 5))
ax.scatter(df['Frame_No'], df['Node3'])  # black lines, semitransparent
plt.tight_layout()
plt.show()

print()
