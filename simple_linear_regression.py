# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as seabornInstance 
from sklearn import metrics
from sklearn.externals import joblib
%matplotlib inline

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')

#explore data by checking number of rows & columns
dataset.shape

#let us get statistical details of data
dataset.describe()

#let us check if there are any nulls
dataset.isnull().any()

#divide the data into “attributes” and “labels”
X = dataset.iloc[:, :-1].values   # attributes X = dataset['YearsExperience'].values
y = dataset.iloc[:, 1].values  #labels y = dataset['Salary].values


#let us check the salary distribution
plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(dataset.iloc[:, 1])


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

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# save the model to disk
filename = 'simplelinearregression_salary.sav'
joblib.dump(regressor, filename)


#load from saved model
loaded_regressor = joblib.load(filename)

# Predicting the Test set results
y_pred = loaded_regressor.predict(X_test)

# let us get rsquare and coefficients
r_sq = loaded_regressor.score(X, y)
co_ef_b0 = loaded_regressor.intercept_
co_ef_b1 = loaded_regressor.coef_  #weights


#Check the difference between the actual value and predicted value.
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(25)

#plot the comparison of Actual and Predicted values
df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, loaded_regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, loaded_regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()