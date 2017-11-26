import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('predicted.csv')
predicted = df['predicted_nikkei']
actual = df['actual_nikkei']

plt.plot(predicted)
plt.plot(actual)
plt.show()
