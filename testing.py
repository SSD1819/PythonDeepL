import matplotlib.pyplot as plt
import numpy as np


plt.subplot(2, 1, 1)
fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
bp = ax.boxplot(trainSamp["budget"], showmeans=True)

plt.subplot(2, 1, 2)
plt.hist(trainSamp["budget"])
plt.show()

plt.hist(trainSamp["popularity"])
