# An-example-for-Machine-Learning-to-Predict-TWII

This is a simple example for machine learning to predict TWII future price step by step. It’s my note when I just learn for half year in university, so if you find any question or have some suggestion, please feel free to tell me. 

#### Objective:
> Using listed stock and TWII close price on t, t-1, … , t-4 making features, to predict TWII close price on t+5.

* Train set: 2010-01-01 ~ 2018-12-31
* Test set: After 2019-01-01
* Feature 1: TWII close price
* Feature 2: Listed stock close price with PCA
* Feature 3: TWII and listed stock close price with PCA

Using GridSearchCV to find best hyperparameter to avoid overfitting. The executing process is on Pipeline 1~6, including Features Engineering, Principal Component Analysis(components = 3 or 5), MinMaxScaler or StandardScaler, Lasso Regression with different Alpha parameter. Thus, Cross-validation use TimeSeriesSplit on sklearn, and the score set “neg mean squared error”.

Enjoy it.
