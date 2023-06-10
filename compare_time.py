import matplotlib.pyplot as plt
import numpy as np


# Execution times in ms for both algorithms
# Both are executed on a image 'test8.jpg'
algo1_times = [26.09, 22.07]
algo2_times = [37.12, 30.17]

# Image labels
labels = ['1st run on test image', '2nd run on test image']

x_pos = np.arange(len(labels))

# Set title and labels and plot bars for both algorithms
plt.xticks(x_pos, labels)
plt.xlabel('Image')
plt.ylabel('Execution Time (In milliseconds)')
plt.title('Comparison of our Lane Detection algorithms')
plt.bar(x_pos - 0.2, algo1_times, 0.25, label='Algorithm 1')
plt.bar(x_pos + 0.2, algo2_times, 0.25, label='Algorithm 2')
plt.legend(loc='best')

plt.show()
