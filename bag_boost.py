import pandas as pd
import numpy as np

#+++++++++++++++++++++++++++ Random Forest ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#https://towardsdatascience.com/random-forest-in-python-24d0893d51c0


# Pandas is used for data manipulation
import pandas as pd
# Read in data and display first 5 rows
features = pd.read_csv('temps.csv')
features.head(5)

print('The shape of our features is:', features.shape)

# Descriptive statistics for each column
print(features.describe())

# One-hot encode the data using pandas get_dummies
features = pd.get_dummies(features)
# Display the first 5 rows of the last 12 columns
print(features.iloc[:,5:].head(5))












#+++++++++++++++++++++++++++++++++++++++ GBM ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d







#++++++++++++++++++++++++++++++++++++++++++++++++++ xgboost ++++++++++++++++++++++++++++++++++++++++++++++++++
#https://www.kaggle.com/nschneider/gbm-vs-xgboost-vs-lightgbm

















#+++++++++++++++++++++++++++++++++++++++++++++++++ LIghtGBM ++++++++++++++++++++++++++++++++++++++++++++++++++








#++++++++++++++++++++++++++++++++++++++++++++ CatBoost ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++




#++++++++++++++++++++++++++++++++++++++++++ AdaBoost +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++






