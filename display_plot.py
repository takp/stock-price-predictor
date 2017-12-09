import pandas as pd
import matplotlib.pyplot as plt

# Plot Training Curve
history_df = pd.read_csv('training_history.csv')
print(history_df.head(10))
training_loss = history_df['loss']
validation_loss = history_df['val_loss']

plt.plot(training_loss)
plt.plot(validation_loss)
plt.show()

# Plot Predicted Result
df = pd.read_csv('predicted.csv')
print(df.head(10))
predicted = df['predicted_nikkei']
actual = df['actual_nikkei']

plt.plot(predicted)
plt.plot(actual)
plt.show()
