# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as seabornInstance 
from sklearn import metrics
from sklearn.externals import joblib
%matplotlib inline


# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')

#explore data by checking number of rows & columns
dataset.shape

#let us get statistical details of data
dataset.describe()

#let us check if there are any nulls
dataset.isnull().any()

X = dataset.iloc[:, :-1].values
y = dataset['Profit'].values

#let us check the profit distribution
plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(dataset['Profit'])


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(fit_intercept = True)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

regressor.predict(np.array([0,0, 101913.08, 110594.11,229160.95]).reshape(-1, 5))

# let us get rsquare and coefficients
print('r_sq :', regressor.score(X, y))
print('co_ef_b0 ', regressor.intercept_
co_ef_b1 = regressor.coef_  #weights


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
