import numpy as np
import matplotlib.pyplot as plt

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
file_names = ["SmallOrLarge", "Checksum", "StringLengthsBackwards", "NegativeToZero", "LastIndexOfZero", "MirrorImage"]
plt.figure(figsize=(18, 12))
plt.title("Code Error Rate Over Generations")
for i in range(len(file_names)):
    file_name = file_names[i]
    ferror = file_name + "errors.npy"
    fiter = file_name + "iterations.npy"
    aerror = np.load(ferror)
    aiter = np.load(fiter)

    plt.plot(aiter, aerror, color=colors[i], linewidth=2, label=file_name)
plt.ylim((-0.01, 0.08))
plt.legend(loc=1, prop={'size': 20})
plt.xlabel('Generations', fontsize=16)
plt.ylabel('Code Error Rate', fontsize=16)
plt.show()
