import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load data into DataFrame
data = pd.read_csv("data_20240506_074336.csv")  # Update with your data file path

# Define features and target variable
X = data[['Open', 'Close', 'Volume', 'High', 'Low']]
y = data['Adj Close']  # or any other target variable you want to predict

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Visualize actual vs. predicted
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.xlabel('Sample')
plt.ylabel('Adj Close Price')
plt.title('Actual vs. Predicted Stock Prices')
plt.legend()
plt.show()
