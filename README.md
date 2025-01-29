# Forex Trading Strategy Analysis and Simulation with GPU Training

## Project Overview

This project presents a comprehensive analysis and simulation of a Forex trading strategy using historical EUR/USD data. Leveraging both CPU and GPU-accelerated libraries, the workflow encompasses data preprocessing, feature engineering, model training with XGBoost, performance evaluation, and Monte Carlo simulations to project future trading outcomes. The objective is to develop a robust machine learning model that can predict trading signals (Buy, Sell, Hold) and assess the strategy's effectiveness through detailed performance metrics and simulations.

## Workflow Summary

1. **Importing Libraries and Configurations**
   - Load essential libraries for data processing (`pandas`, `cudf`, `cupy`), modeling (`xgboost`), evaluation (`scikit-learn`), and visualization (`matplotlib`).
   - Configure display settings and verify the RAPIDS cuDF library version.

2. **Defining Parameters and Date Ranges**
   - Set date ranges for training (`2022-01-01` to `2022-12-31`), validation (`2023-01-01` to `2023-05-31`), and testing (`2023-06-01` to `2023-10-31`) datasets.
   - Establish trading hours, profit targets, stop-loss parameters, leverage factors, and simulation settings.

3. **Data Loading and Initial Preprocessing**
   - Load Forex data from a Parquet file into a GPU-accelerated cuDF DataFrame.
   - Filter data based on extended date ranges and specific trading conditions (currency pair: EUR/USD, timeframe: 10 minutes).

4. **Feature Engineering**
   - Generate lagged features to incorporate historical data.
   - Create time-based features (e.g., working hours, day of the week) to capture temporal patterns.
   - Develop target variables (`Y`) based on profit target and stop-loss conditions, labeling trades as Buy (1), Sell (2), or Hold (0).
   - Construct ratio-based features to enhance model input with relationships between different indicators.

5. **Feature Selection and Correlation Analysis**
   - Identify and remove highly correlated features (correlation coefficient > 0.95) to mitigate multicollinearity and improve model performance.

6. **Data Splitting and Scaling**
   - Split the data into training, validation, and test sets based on predefined date ranges.
   - Apply robust scaling to feature matrices to handle outliers and ensure uniform feature contribution.

7. **Model Training with XGBoost**
   - Define hyperparameters tailored for multi-class classification using XGBoost.
   - Train the model with early stopping based on validation loss to prevent overfitting.
   - Visualize training and validation loss progression over boosting rounds.

8. **Model Evaluation**
   - Generate classification reports for training, validation, and test datasets, detailing precision, recall, f1-score, and support for each class.
   - Analyze precision and recall across different probability thresholds to determine optimal classification thresholds for Buy and Sell signals.

9. **Trading Simulation**
   - Simulate trading based on model predictions, tracking correct trades, total trades, and capital progression.
   - Calculate performance metrics such as average daily return, standard deviation, and Sharpe Ratio.
   - Visualize the value progression over time based on trading activities.

10. **Monte Carlo Simulations**
    - Conduct Monte Carlo simulations to project the distribution of final capital after a specified number of years.
    - Analyze simulation results through histograms and capital progression plots to assess potential financial outcomes.

11. **Visualization and Reporting**
    - Create various plots to visualize model performance, trade accuracy over time, precision-recall trade-offs, and simulation distributions.
    - Summarize findings and insights derived from the analysis.

## Objectives

- **Predictive Modeling:** Develop a machine learning model capable of accurately predicting Buy and Sell signals in Forex trading.
- **Performance Evaluation:** Assess the model's effectiveness using classification metrics and analyze trading strategy performance through simulations.
- **Risk Assessment:** Utilize Monte Carlo simulations to understand potential financial risks and returns associated with the trading strategy.
- **Visualization:** Provide clear and informative visualizations to communicate results and support decision-making.

## Tools and Technologies

- **Programming Language:** Python
- **Libraries:** pandas, cudf, cupy, xgboost, scikit-learn, matplotlib, joblib
- **Environment:** GPU-accelerated computing with RAPIDS cuDF for efficient data processing

---
