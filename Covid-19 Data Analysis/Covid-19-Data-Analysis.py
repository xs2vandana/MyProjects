#!/usr/bin/env python
# coding: utf-8

# # Covid-19 Data Analysis
# # Dataset: worldometer_coronavirus_daily_data.csv

# In[738]:


import pandas as pd
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from datetime import date
import datetime as dt
from datetime import timedelta
from datetime import datetime, date
import statistics as st
import scipy.stats
import string


# In[758]:


df = pd.read_csv (r'/Users/vandy/desktop/new_data/worldometer_coronavirus_daily_data.csv',parse_dates=['date'])   #read the csv file (put 'r' before the path string to address any special characters in the path, such as '\'). Don't forget to put the file name at the end of the path + ".csv"
df.tail(5)


# In[759]:


df.columns


# In[760]:


df.shape


# # Checking for null values

# In[761]:


df.isnull().sum()


# # Checking %wise null values

# In[762]:


round(df.isnull().sum(axis=0).sort_values(ascending=False)/len(df)*100,0)


# In[763]:


df.dropna()


# In[764]:


df.describe()


# In[765]:


df['daily_new_cases'].sum()


# In[766]:


df['mnth_yr'] = df['date'].apply(lambda x: x.strftime('%B-%Y')) 
df.head(2)


# # Checking & visualizing top10 counties with the most coronavirus daily cases

# In[767]:


df_countrywise=df.groupby('country')['daily_new_cases'].sum().sort_values(ascending=False).head(10)
df_countrywise


# In[768]:


df_countrywise.plot.bar()
plt.show()


# # Separating US data (on which i am making an analysis on daily cases )

# In[769]:


US=df[df.country=="USA"]
US.tail(5)


# In[770]:


US['daily_new_cases'].sum()


# In[771]:


US.isnull().sum()


# # Dropping the row which contains NULL value in "daily_new_cases" ( dropping as i am conducting my analysis on this column  

# In[772]:


US=US.drop([80280])


# In[773]:


US.isnull().sum()


# In[774]:


nn=US.groupby('mnth_yr')['daily_new_cases'].sum()
nn


# In[775]:


#US['date']=US['date'].map(dt.datetime.toordinal)


# In[776]:


US.tail()


# # Checking cumulative total cases in US

# In[777]:


print("basic info")
print("total no of cases",US['cumulative_total_cases'].iloc[-1])
print("total no of deaths",US['cumulative_total_deaths'].iloc[-1])


# In[778]:


US_daily_cases=US.groupby('date')['daily_new_cases'].sum().sort_values()
US_daily_cases


# In[779]:


boxplot = US.boxplot(column=['daily_new_cases'])


# In[780]:


plt.hist(US['daily_new_cases'])


# # Ploting a Line chart to see the daily trend

# In[781]:


sns.set(rc={'figure.figsize':(15,10)})
sns.lineplot(x="date",y="daily_new_cases",data=US)
plt.show


# # Creating Model Using Regression

# In[799]:


from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import randomizedSearchCV,train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


# In[800]:


US.tail(2)


# In[801]:


x=US['date']
y=US['daily_new_cases']


# In[802]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[803]:


#lr.fit(np.array(x_train).reshape(-1,1),np.array(y_train).reshape


# In[804]:


x=x.values.reshape(-1,1)


# In[805]:


y=y.values.reshape(-1,1)


# In[806]:


from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
sx=sc_x.fit_transform(x)
sy=sc_y.fit_transform(y)


# In[807]:


from sklearn.svm import SVR
reg=SVR(kernel='rbf')
#lr.fit(np.array(x_train).reshape(-1,1),np.array(y_train).reshape
reg.fit(np.array(x_train).reshape(-1,1),np.array(y_train).reshape(-1,1))
#reg.fit(sx,sy)


# In[808]:


y_pred = reg.predict(x_test.values.reshape(-1,1))
y_pred = sc_y.inverse_transform(y_pred)


# # Step 7: Comparing the Test Set with Predicted Values

# In[809]:


df_p = pd.DataFrame({'Real Values':sc_y.inverse_transform(y_test.values.reshape(-1)), 'Predicted Values':y_pred})
df_p


# In[817]:


reg.fit(sx,sy)


# In[818]:


reg.score(sx,sy)*100


# In[819]:


reg.predict(np.array([[737854]]))


# In[820]:


plt.scatter(sx,sy)
plt.plot(sx,reg.predict(sx),color='k')
plt.show()


# In[796]:


reg.predict(np.array([[737854]]))


# In[641]:


from sklearn.ensemble import RandomForestRegressor
regf=RandomForestRegressor(n_estimators=100)
regf.fit(x,y)


# In[642]:


regf.score(x,y)


# In[643]:


plt.scatter(x,y)
plt.plot(x,regf.predict(x),color='k')
plt.show()


# In[ ]:




