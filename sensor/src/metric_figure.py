import matplotlib.pyplot as plt

# Create a figure and an axes object
fig, ax = plt.subplots()

# Plot the data
ax.plot([0, 10, 100, 1000, 10000, 100000], [0, 0.093603729, 0.116296696, 0.465304116, 2.074620929, 3.189112756], color='red', label='chamfer')
ax.plot([0, 10, 100, 1000, 10000, 100000], [0, 0.085920267, 0.106750482, 0.427109628, 1.498048789, 1.66891853], color='blue', label='gaussian')

# Set the x-axis and y-axis labels
ax.set_xlabel('T_bias',fontdict = {'size':28, 'family':'Times New Roman'})
ax.set_ylabel('Metric',fontdict = {'size':28, 'family':'Times New Roman'})
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)

# Add a title to the plot
ax.set_title('metric change with T_bias',fontdict = {'size':28, 'family':'Times New Roman'})
# Set the x-axis to a logarithmic scale
ax.set_xscale('log')
# Add a legend to the plot
ax.legend(fontsize=24)
# # Show the plot
# plt.show()
# plt.clf()

# Create a figure and an axes object
fig2, ax2 = plt.subplots()

# Plot the data
ax2.plot([0, 10, 100, 1000, 10000, 100000], [0, 0.000209886, 0.00197271, 0.033201909, 0.196596181, 1.459193458], color='red', label='chamfer')
ax2.plot([0, 10, 100, 1000, 10000, 100000], [0, 2.85E-05, 0.00031591, 0.01524162, 0.042270326, 0.248062504], color='blue', label='gaussian')
# Set the x-axis and y-axis labels
ax2.set_xlabel('Noise',fontdict = {'size':28, 'family':'Times New Roman'})
ax2.set_ylabel('Metric',fontdict = {'size':28, 'family':'Times New Roman'})
ax2.tick_params(axis='x', labelsize=18)
ax2.tick_params(axis='y', labelsize=18)
# Add a title to the plot
ax2.set_title('metric change vs Noise',fontdict = {'size':28, 'family':'Times New Roman'})

# Set the x-axis to a logarithmic scale
ax2.set_xscale('log')

# Add a legend to the plot
ax2.legend(fontsize=24)

# Show the plot
plt.show()