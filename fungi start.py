import pandas as pd
import matplotlib.pyplot as plt

# Load CSV data
df = pd.read_csv('test2.csv', index_col=0)

df.index = pd.to_timedelta(df.index)

# Plot the data
plt.figure(figsize=(14, 8))
for column in df.columns:
    plt.plot(df.index.total_seconds(), df[column], label=column)

plt.title('PicoLog Data')
plt.xlabel('Time (seconds)')
plt.ylabel('Microvolts (µV)')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()

plt.show()