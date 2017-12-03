import numpy as np
import pylab as plt
import pandas as pd

data = pd.read_csv('data/data.csv')
for feature in ['nikkei', 'nasdaq', 'currency']:
    dataset = data[feature]
    print("[{}] Mean: {}".format(feature, np.mean(dataset)))
    print("[{}] Standard deviation: {}".format(feature, np.std(dataset)))
    plt.xlabel(feature, fontsize=18)
    plt.hist(dataset, normed=True, bins=50)
    plt.show()
