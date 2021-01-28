# An-example-for-Machine-Learning-to-Predict-TWII

This is a simple example for machine learning to predict TWII future price step by step. It’s my note when I just learn for half year in university, so if you find any question or have some suggestion, please feel free to tell me. Also this is my first GitHub project!

#### Objective:
> Using listed stock and TWII close price on t ~ t-4 making features(close price and mean average), to predict TWII close price on t+5.

* Train set: 2010-01-01 ~ 2018-12-31
* Test set: After 2019-01-01
* Feature 1: TWII close price
* Feature 2: Listed stock close price with PCA
* Feature 3: TWII and listed stock close price with PCA

On GridSearchCV, it would find the best hyperparameter to avoid overfitting. The executing process is standardized on Pipeline 1~6, including Features Choosing, Principal Component Analysis(components = 3 or 5), MinMaxScaler or StandardScaler, Lasso Regression with different Alpha parameter. Besides, Cross-validation use TimeSeriesSplit on sklearn, and the score set “neg mean squared error”.

Final result about train test predction show on pictures, also there is one picture compare with different Alpha. Downlaod the code and file you can do by yourself. Just change the file adress. 

Hope you will like it.

Enjoy.
