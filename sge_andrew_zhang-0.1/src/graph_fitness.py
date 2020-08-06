import numpy as np
import matplotlib.pyplot as plt

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
file_names = ["LastIndexOfZero"]
plt.figure(figsize=(18, 12))
plt.title("Top Average Fitness for Last Index of Zero Over Generations")
for i in range(len(file_names)):
    file_name = file_names[i]
    fitness = file_name + "fitness.npy"
    fiter = file_name + "iterations.npy"
    afitness = np.load(fitness)
    aiter = np.load(fiter)

    plt.plot(aiter, afitness, color=colors[i], linewidth=2, label=file_name)
# plt.ylim((-0.1, 1.0))
plt.legend(loc=1, prop={'size': 20})
plt.xlabel('Generations', fontsize=16)
plt.ylabel('Fitness', fontsize=16)
plt.show()
