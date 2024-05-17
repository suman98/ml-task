import numpy as np
import matplotlib.pyplot as plt

# Assuming df is your DataFrame containing necessary data

# Calculate Predicted Signal
df['Predicted_Signal'] = model.predict(X)

# Calculate AAPL Returns
df['AAPL_returns'] = np.log(df['Close']/df['Close'].shift(1))

# Calculate Cumulative AAPL Returns
Cumulative_AAPL_returns = np.cumsum(df[split:]['AAPL_returns'])

# Calculate Strategy Returns
df['Strategy_returns'] = df['AAPL_returns'] * df['Predicted_Signal'].shift(1)

# Calculate Cumulative Strategy Returns
Cumulative_Strategy_returns = np.cumsum(df[split:]['Strategy_returns'])

# Plotting
plt.figure(figsize=(10,5))

# Plot Cumulative AAPL Returns
plt.plot(Cumulative_AAPL_returns, color='b', label='Cumulative Actual Returns')

# Plot Cumulative Strategy Returns
plt.plot(Cumulative_Strategy_returns, color='g', label='Cumulative Strategy Returns')

# Plot Predicted Signals
plt.plot(df.index[split:], df['Predicted_Signal'][split:], 'r.', label='Predicted Signal')

plt.legend()
plt.show()
