import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

data=pd.read_csv('Rainfall_1901_2016_PAK.csv')
data.columns

data.rename(columns={'Rainfall - (MM)':'Rainfall-(mm)',' Year':'Year'},inplace=True)
data.columns

print(data.head(10))

# Through a bar graph, the average rainfall for each year is represented on the x-axis, 
# while the recorded rainfall (in mm) for that year is shown on the y-axis. 
# This allows you to observe temporal rainfall trends and variations between different years. 
# The height of each bar in the graph represents the amount of rainfall, enabling you to compare and understand 
# the differences in rainfall between different years.
plt.figure(figsize=(15, 8))  # Set the figure size

filtered_data = data.loc[data['Year'] >= 1990]  # Filter data for years >= 1990

plt.bar(filtered_data['Year'], filtered_data['Rainfall-(mm)'])
plt.xlabel('Year')
plt.ylabel('Rainfall (mm)')
plt.title('Climate Change Rainfall During 2000 to 2016 ')

plt.show()

import matplotlib.pyplot as plt

filtered_data = data.loc[(data['Year'] >= 1980) & (data['Year'] <= 2000)]  # Filter data for years 1980 to 2000

fig, ax = plt.subplots(figsize=(10, 8))  # Create a figure and axis object

colors = filtered_data['Month'].unique()  # Get unique colors based on months

for color in colors:
    subset_data = filtered_data[filtered_data['Month'] == color]
    ax.plot(subset_data['Year'], subset_data['Rainfall-(mm)'], label=color)

ax.set_xlabel('Year')
#ax.set_ylabel('Rainfall (mm)')
np.random.seed(0)
X = np.random.rand(100, 1) * 10
y = 2 * X + np.random.randn(100, 1)
ax.set_title('Climate Change Disaster by Rainfall in Year 1980 to 2000')
ax.legend()

plt.show()

import matplotlib.pyplot as plt

sel = data.loc[(data['Year'] >= 1980) & (data['Year'] <= 2000) & (data['Month'] == "June")]

fig, ax = plt.subplots(figsize=(10, 8))  # Create a figure and axis object

colors = sel['Month'].unique()  # Get unique colors based on months

for color in colors:
    subset_data = sel[sel['Month'] == color]
    ax.plot(subset_data['Year'], subset_data['Rainfall-(mm)'], label=color)

ax.set_xlabel('Year')
#ax.set_ylabel('Rainfall (mm)')
ax.set_title('Climate Change in June Year 1980 to 2000')
ax.set_xlim([1979, 2001])
ax.legend()

for i, txt in enumerate(sel['Rainfall-(mm)']):
    ax.annotate(txt, (sel['Year'].iloc[i], sel['Rainfall-(mm)'].iloc[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.show()


for i, txt in enumerate(sel['Rainfall-(mm)']):
    ax.annotate(txt, (sel['Year'].iloc[i], sel['Rainfall-(mm)'].iloc[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.show()
sel1 = data.loc[(data['Year'] >= 2001) & (data['Year'] <= 2016) & (data['Month'] == "July")]

fig1, ax1 = plt.subplots(figsize=(10, 8))  # Create a figure and axis object

colors1 = sel1['Month'].unique()  # Get unique colors based on months

for color in colors1:
    subset_data = sel1[sel1['Month'] == color]
    ax1.plot(subset_data['Year'], subset_data['Rainfall-(mm)'], label=color)

ax1.set_xlabel('Year')
ax1.set_ylabel('Rainfall (mm)')
ax1.set_title('Climate Change in July during Year 2001 to 2016')
ax1.set_xlim([2000, 2017])
ax1.legend()

for i, txt in enumerate(sel1['Rainfall-(mm)']):
    ax1.annotate(txt, (sel1['Year'].iloc[i], sel1['Rainfall-(mm)'].iloc[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.show()


import matplotlib.pyplot as plt

sel = data.loc[(data['Year'] >= 1980) & (data['Year'] <= 2000) & (data['Month'] == "August")]

fig, ax = plt.subplots(figsize=(10, 8))  # Create a figure and axis object

colors = sel['Month'].unique()  # Get unique colors based on months

for color in colors:
    subset_data = sel[sel['Month'] == color]
    ax.plot(subset_data['Year'], subset_data['Rainfall-(mm)'], label=color)

ax.set_xlabel('Year')
#ax.set_ylabel('Rainfall (mm)')
ax.set_title('Climate Change in August Year 1980 to 2000')
ax.set_xlim([1979, 2001])
ax.legend()

for i, txt in enumerate(sel['Rainfall-(mm)']):
    ax.annotate(txt, (sel['Year'].iloc[i], sel['Rainfall-(mm)'].iloc[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.show()


sel1 = data.loc[(data['Year'] >= 2001) & (data['Year'] <= 2016) & (data['Month'] == "August")]

fig1, ax1 = plt.subplots(figsize=(10, 8))  # Create a figure and axis object

colors1 = sel1['Month'].unique()  # Get unique colors based on months

for color in colors1:
    subset_data = sel1[sel1['Month'] == color]
    ax1.plot(subset_data['Year'], subset_data['Rainfall-(mm)'], label=color)

ax1.set_xlabel('Year')
ax1.set_ylabel('Rainfall (mm)')
ax1.set_title('Climate Change in August during Year 2001 to 2016')
ax1.set_xlim([2000, 2017])
ax1.legend()

for i, txt in enumerate(sel1['Rainfall-(mm)']):
    ax1.annotate(txt, (sel1['Year'].iloc[i], sel1['Rainfall-(mm)'].iloc[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.show()
import matplotlib.pyplot as plt

sel = data.loc[data['Year'] >= 2000, ['Year', 'Rainfall-(mm)']]

year_counts = sel['Year'].value_counts()
rainfall_values = sel.groupby('Year')['Rainfall-(mm)'].sum()

fig, ax = plt.subplots(figsize=(8, 6))  # Create a figure and axis object

ax.pie(rainfall_values, labels=rainfall_values.index, autopct='%1.1f%%', startangle=90)

ax.set_title('Number of Percent Disaster by climate change with Rainfall Year during 2000 to 2016')

plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

plt.show()
import numpy as np
import matplotlib.pyplot as plt





# Apply KMeans clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Get cluster centers
cluster_centers = kmeans.cluster_centers_

# Fit Linear Regression
regression = LinearRegression()
regression.fit(X, y)

# Predict using Linear Regression
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred = regression.predict(X_test)

# Plot the data points and cluster centers
plt.scatter(X, y, c=kmeans.labels_, cmap='viridis')
plt.scatter(cluster_centers[:, 0], regression.predict(cluster_centers), c='red', marker='x', label='Cluster Centers')

# Plot the linear regression line
plt.plot(X_test, y_pred, color='black', linewidth=2, label='Linear Regression')

plt.xlabel('X')
plt.ylabel('y')
plt.title('Cluster of disasters using KMeans Clustering and Prediction by Linear Regression')
plt.legend()
plt.show()
