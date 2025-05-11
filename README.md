# Stock Price Prediction using XGBoost

A machine learning project that uses XGBoost regression to predict stock prices based on technical indicators. This model analyzes historical price data and creates predictions for future stock movements.

![Stock Price Prediction](https://via.placeholder.com/800x400?text=Stock+Price+Prediction+Model)

## Features

- **Historical Price Analysis:** Process and visualize OHLC (Open, High, Low, Close) stock data
- **Technical Indicators Generation:**
  - Multiple Moving Averages (EMA 9/22/35, SMA 5/10/15/30)
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
- **XGBoost Model:** Fine-tuned using GridSearchCV for optimal hyperparameter selection
- **Performance Visualization:** Compare predicted vs. actual prices in interactive plots

## Project Structure

The project is organized as a Jupyter notebook with the following components:

1. **Data Loading & Preprocessing**
   - Import historical stock data
   - Convert dates and set up time series formatting
   - Visualize raw price and volume data

2. **Feature Engineering**
   - Calculate exponential and simple moving averages
   - Generate RSI indicator
   - Calculate MACD and signal line
   - Prepare data for prediction by shifting target values

3. **Model Training**
   - Split data into train/validation/test sets
   - Hyperparameter tuning with GridSearchCV
   - Train XGBoost model with optimal parameters

4. **Evaluation & Visualization**
   - Assess feature importance
   - Calculate prediction error metrics
   - Visualize predicted prices against actual values

## Technologies Used

- **Python 3:** Primary programming language
- **Pandas:** Data manipulation and analysis
- **NumPy:** Numerical computations
- **XGBoost:** Gradient boosting machine learning library
- **Scikit-learn:** Model evaluation and hyperparameter tuning
- **Plotly:** Interactive data visualization
- **Matplotlib:** Static visualization for feature importance

## Requirements

- Python 3.6+
- pandas
- numpy
- xgboost
- scikit-learn
- matplotlib
- plotly

## Installation

1. Clone this repository
