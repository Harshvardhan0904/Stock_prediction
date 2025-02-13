Netflix Stock Price Prediction using Linear Regression

Overview

This project predicts Netflix's stock prices using a Linear Regression model. It takes historical stock price data as input and forecasts future prices based on trends.

Dataset

The dataset consists of historical Netflix stock prices.

It includes columns such as Date, Open, High, Low, Close, Volume.

The closing price is used as the target variable for prediction.

Technologies Used

Python

Pandas

NumPy

Scikit-Learn (Linear Regression)

Matplotlib & Seaborn (Data Visualization)


Methodology

Data Preprocessing:

Load the dataset and handle missing values.

Convert date columns to datetime format.

Extract relevant features.

Exploratory Data Analysis (EDA):

Visualize stock price trends.

Identify correlations between features.

Model Training:

Split the data into training and testing sets.

Train a Linear Regression model on historical data.

Prediction & Evaluation:

Predict stock prices using the trained model.

Evaluate the model using RMSE (Root Mean Square Error).

Visualization:

Plot actual vs. predicted stock prices to analyze model performance.

Results

The model provides a basic prediction of Netflix stock trends.

Accuracy depends on historical data patterns and external market conditions.

Future Improvements

Use more advanced models such as LSTM (Long Short-Term Memory) or Random Forest Regression.

Incorporate external factors like news sentiment, market indices, and earnings reports.

