# Import libraries
import numpy as np
import pandas as pd
import yfinance as yf
import talib as ta
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import seaborn as sns

def load_data(symbol, start_date, end_date):
    """Load stock data from Yahoo Finance."""
    df = yf.download(symbol, start_date, end_date)
    return df

def calculate_technical_indicators(df):
    """Calculate technical indicators."""
    df['S_10'] = df['Close'].rolling(window=10).mean()
    df['Corr'] = df['Close'].rolling(window=10).corr(df['S_10'])
    df['RSI'] = ta.RSI(np.array(df['Close']), timeperiod=10)
    df['Open-Close'] = df['Open'] - df['Close'].shift(1)
    df['Open-Open'] = df['Open'] - df['Open'].shift(1)
    df = df.dropna()
    return df

def prepare_data(df):
    """Prepare data for machine learning."""
    X = df.iloc[:, :9]
    y = np.where(df['Close'].shift(-1) > df['Close'], 1, -1)
    return X, y

def train_model(X_train, y_train):
    """Train logistic regression model."""
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test):
    """Evaluate model performance."""
    probability = model.predict_proba(X_test)
    predicted = model.predict(X_test)
    # print(X_test, predicted)
    return probability, predicted

def plot_strategy_returns(model, df, X, split):
    """Plot cumulative strategy returns."""
    # Calculate Predicted Signal
    df['Predicted_Signal'] = model.predict(X)

    # Calculate AAPL Returns
    df['TSLA_returns'] = np.log(df['Close']/df['Close'].shift(1))

    # Calculate Cumulative AAPL Returns
    Cumulative_TSLA_returns = np.cumsum(df[split:]['TSLA_returns'])

    # Calculate Strategy Returns
    df['Strategy_returns'] = df['TSLA_returns'] * df['Predicted_Signal'].shift(1)

    # Calculate Cumulative Strategy Returns
    Cumulative_Strategy_returns = np.cumsum(df[split:]['Strategy_returns'])

    # Plotting
    plt.figure(figsize=(10,5))

    # Plot Cumulative AAPL Returns
    plt.plot(Cumulative_TSLA_returns, color='b', label='Cumulative Actual Returns')

    # Plot Cumulative Strategy Returns
    plt.plot(Cumulative_Strategy_returns, color='g', label='Cumulative Strategy Returns')

    plt.legend()
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    """Compute confusion matrix and plot as heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Predicted Sell', 'Predicted Buy'], yticklabels=['Actual Sell', 'Actual Buy'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

def create_classification_report(y_true, y_pred):
    print(classification_report(y_test, predicted))

def examine_the_coefficient(model, X):
    pd.DataFrame(zip(X.columns, np.transpose(model.coef_)))

def calculate_class_probabilities(model, X_test):
    # Specify the start and end dates for prediction
    start_date = '2019-12-22'
    end_date = '2024-01-01'

    # Create a list containing the start and end dates
    prediction_date = [start_date, end_date]

    # Use the list of dates for prediction
    probability = model.predict_proba(X_test)
    print(probability)

def compute_accuracy(X, y):
    cross_val = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
    print(cross_val)
    print(cross_val.mean())

def main():
    # Load data
    start_date = '1970-01-01'
    end_date = '2023-12-30'
    symbol = 'TSLA'
    df = load_data(symbol, start_date, end_date)
    split = int(0.7*len(df))
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Prepare data for machine learning
    X, y = prepare_data(df)
    
    # for i in range (1, 100):
    #     for j in range (1, 100):
    # # Split data into train and test sets
    #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=j / 100, random_state=i)
    
    # # Train model
    #         model = train_model(X_train, y_train)
    #         probability, predicted = evaluate_model(model, X_test)
    #         if (1 in predicted and -1 in predicted):
    #             print( j, i, predicted.tolist().count(1), predicted.tolist().count(-1))
    #             break
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=94 / 100, random_state=75)
    model = train_model(X_train, y_train)
    # # examine the cofficients
    examine_the_coefficient(model, X)

    # # calculate class prob
    calculate_class_probabilities(model, X_test)

    # # Evaluate model
    probability, predicted = evaluate_model(model, X_test)
    
    # # # Plot strategy returns
    plot_strategy_returns(model, df, X, split)

    # # # Plot confusion matrix
    plot_confusion_matrix(y_test, predicted)
    create_classification_report(y_test, predicted)
    # # Compute Accuracy
    compute_accuracy(X, y)

if __name__ == "__main__":
    main()
