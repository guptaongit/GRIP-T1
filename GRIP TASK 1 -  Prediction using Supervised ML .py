#!/usr/bin/env python
# coding: utf-8

# # GRIP - DATA SCIENCE AND BUSINESS ANALYTICS #TASK 1
# 
# 
# # Prediction using Supervised ML 
# 
# Predict the percentage of an student based on the no. of study hours.                                   
#                                                                                                              SHIVAM GUPTA

# In[1]:


#LETS IMPORT THE REQUIRED LIBRARIES
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')
#HERE,READING DATA FROM THE REMOTE LINK
url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Data imported successfully From the remote link")
data.head(10)


# In[2]:


#Now lets Split the data into training and test sets
X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values  
from sklearn.model_selection import train_test_split  
X_t, X_ts, y_t, y_ts = train_test_split(X, y,test_size=0.2, random_state=0)


# In[3]:


#Training the algorithm
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_t, y_t) 

print("Training Successful.")


# In[11]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line,'r');
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score') 
plt.grid()
plt.show()


# Here we observe positive a positive linear relationship between number of hours studied and the score received.

# In[5]:


print(X_ts) # Testing data - In Hours
y_pred = regressor.predict(X_ts) # Predicting the scores
# Comparing Actual vs Predicted
data_frame = pd.DataFrame({'Actual Value': y_ts, 'Predicted Value': y_pred})  
data_frame


# In[6]:


#Predicting score for 8 hours
hours = 8
own_pred = regressor.predict([[hours]])
print("Number of Hours are = {}".format(hours))
print("Predicted Score is  = {}".format(own_pred[0]))


# In[7]:


#Evaluating the model
from sklearn import metrics  
print('Mean Absolute Error is :', 
      metrics.mean_absolute_error(y_ts, y_pred))


# Hence,sucessfully completed task : Predict the percentage of an student based on the no. of study hours. 
# 
