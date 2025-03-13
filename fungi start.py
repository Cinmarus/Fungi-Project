import pandas as pd
import matplotlib.pyplot as plt

# Load CSV data
df = pd.read_csv('test2.csv', index_col=0)

# Convert time index to a format matplotlib understands (optional)
df.index = pd.to_timedelta(df.index)

# Plot the data
plt.figure(figsize=(14, 8))
for column in df.columns:
    plt.plot(df.index.total_seconds(), df[column], label=column)

# Add plot labels and title
plt.title('PicoLog Data')
plt.xlabel('Time (seconds)')
plt.ylabel('Microvolts (µV)')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()

# Display the graph
plt.show()