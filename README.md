## FoodPriceBD – Bangladesh Essential Food Price Volatility Analysis

FoodPriceBD is a data science project that analyzes and predicts essential food price volatility in Bangladesh. The project focuses on common food commodities such as rice, wheat flour, lentils, oil, sugar, potato, onion, chicken, fish, and eggs across different divisions of Bangladesh.

The main goal of this project is to understand food price trends, seasonal patterns, regional price differences, and future price movement. The project uses real-world market data collected through web scraping and APIs. When live data is unavailable, a realistic synthetic fallback dataset is generated to continue the analysis.

This project includes data collection, data cleaning, exploratory data analysis, feature engineering, regression-based price prediction, and classification-based price movement detection. Regression models are used to predict food prices, while classification models are used to classify price movement as Increase, Decrease, or Stable.

### Key Features

- Collects food price data using web scraping and APIs
- Uses a realistic synthetic fallback dataset when live data is unavailable
- Performs exploratory data analysis on food price trends
- Analyzes commodity-wise, region-wise, monthly, and yearly price patterns
- Creates visualizations such as histograms, boxplots, heatmaps, trend plots, and comparison charts
- Builds regression models for food price prediction
- Builds classification models for price movement detection
- Compares multiple machine learning models using evaluation metrics
- Generates feature importance and confusion matrix visualizations

### Dataset Overview

The project dataset contains more than 10,000 observations. It covers:

- 10 essential food commodities
- 8 divisions of Bangladesh
- Monthly price records
- Commodity, market, region, date, and price information

### Machine Learning Models Used

#### Regression Models

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regression

#### Classification Models

- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Machine
- Neural Network

### Technology Stack

- R Programming Language
- dplyr
- tidyr
- ggplot2
- caret
- randomForest
- rpart
- glmnet
- gbm
- e1071
- nnet
- httr
- rvest
- jsonlite

### Project Objective

The objective of this project is to apply data science techniques to analyze essential food price volatility in Bangladesh and build predictive models that can help understand future price behavior. This type of analysis can support market monitoring, food security research, and data-driven decision-making.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Author

Md. Kamrul Hasan
