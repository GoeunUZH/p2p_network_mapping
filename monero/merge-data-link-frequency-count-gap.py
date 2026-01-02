import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning


# Load data
data = pd.read_csv('data/merged_20241221_datawith3nodes_no_timestamp.txt', sep=',', header=0, usecols=[0, 1, 2])
data = data.iloc[1:]
data.columns = ['ip1', 'ip2', 'count']

# Ensure 'count' is numeric, coercing errors to NaN
data['count'] = pd.to_numeric(data['count'], errors='coerce')

# Remove rows where 'count' is NaN or less than or equal to 1
data = data.dropna(subset=['count'])
data = data[data['count'] > 1]
MIN_CONNECTIONS = 7

# Group data by 'ip1' and calculate sum for each group
groupd = data.groupby('ip1')
for i, g in groupd:
    ip = g.iloc[0]['ip1']
    s = np.sum(np.unique(list(g['count'])))
    data.loc[data['ip1'] == ip, 'sum'] = s

data['y'] = 0
data['x'] = data['count'] / data['sum']
data = data.sort_values(by=['ip1', 'count'], ascending=[0, 0])

# Group again by 'ip1' to process individual IPs
grouped = data.groupby('ip1')

for group_name, group_data in grouped:
    if len(group_data) > 7:
        c = np.array(group_data['count'])
        if c[0] != c[-1]:
            x = np.unique(np.array(list(group_data['count'])))
            x = np.sort(x)[::-1] / np.sum(x)

            y = np.zeros(len(x))

            data1 = list(zip(x, y))
            kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(data1)
            label = kmeans.labels_
            if label[0] == 0:
                label = abs(label - 1) * 1.0
            index = np.where(label == 1)[-1][-1]

            labels = np.array([1] * (index + 1) + [0] * (len(group_data) - index - 1))

            data.loc[data['ip1'] == group_name, 'label'] = labels
        else:
            data.loc[data['ip1'] == group_name, 'label'] = "="
    else:
        data.loc[data['ip1'] == group_name, 'label'] = "<7"

valid_data = data[data['label'] != '<7']
valid_data = valid_data[valid_data['label'] != '=']

valid_data = valid_data.sort_values(by=['ip1', 'count'], ascending=[0, 0])
grouped = valid_data.groupby('ip1')

minx = []
minx_ave = []
maxx = []
maxx_ave = []
maxc = []
for gn, gd in grouped:
    x = np.array(gd['x'])
    index = np.where(np.array(gd['label']) == 1)[-1][-1]
    minx_ave.append(np.average(np.array(x[:index + 1])))

    minx.append(x[index])
    maxx.append(x[index + 1])
    maxx_ave.append(np.average(np.array(x[index + 1:])))
    maxc.append(gd.iloc[0]['count'])

# Save data where label is 1
save_data = valid_data[valid_data['label'] == 1]
print(len(save_data))
print(len(np.unique(save_data['ip1'])))
save_data.to_csv('data/processed_merge_20241221_data_3weeks_with3nodes.txt')

# Plot the result
plt.figure(figsize=(12, 8))
plt.scatter(np.array(minx) - np.array(maxx), np.array(minx_ave) - np.array(maxx_ave),color='#0028A5',alpha=0.5)
plt.xlabel(r'$\mathrm{min}_{high} - \mathrm{max}_{low}$', fontsize=20)  # Modify x-axis
plt.ylabel(r'$\mathrm{ave}_{high} - \mathrm{ave}_{low}$', fontsize=20)  # Modify y-axis
plt.title('Quality of interaction data filtering by K-means clustering algorithm', fontsize=18)
plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), 'r')
plt.xscale('log')
plt.yscale('log')
plt.grid(True)

# Add horizontal line y=0
plt.axhline(y=0, color='r', linestyle='--')
plt.ylim(0.01, 1)
plt.xlim(0.01, 1)
plt.tight_layout()

# Save the plot as an image file
plt.savefig('pic/gap_20241211_3nodes_plot_new.png')  # Save the figure as PNG file
# plt.show()
