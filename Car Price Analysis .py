#!/usr/bin/env python
# coding: utf-8

# # Car price Dataset (from kaggle)
# Performed: 1) Data collecting 
#            2) Data Sanitization 
#            3) Problem Statements 
#            4) EDA
#            5) Model Building 
#            6) Prediction 
#            7) Regression Plot(Actual vs Predicted Values)

# # Problem Statements:
# The questions are taken from Google and HackerRank both combined.
# 
# 1) Drop all the cars where the price < 10000.
# 2) Which car was bought maximum number of times by customers and whats the highest price for a car in the dataset?
# 3) Sort and Return the resultant dataframe in acending order by price.
# 4) Does Fuel Type affecting the price of cars differently?
# 5) Which variables are significant in predicting the price of a car. 
# 6) How well those variables describe the price of a car.

# In[1]:


#importing dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import os


# # Data collecting

# In[2]:


#loading dataset
df= pd.read_csv('CarPrice_Assignment.csv')


# In[3]:


df.head(5)


# In[4]:


df.shape


# Dataset contains of 205 rows & 26 columns.

# In[5]:


df.info()


# # Data Sanitization

# In[6]:


#checking for null values
df.isnull().sum()


# No null values were present.

# In[7]:


#checking for duplicated values
df.duplicated().sum()


# No duplicate values were present.

# # Problem Statements

# # 1. Drop all the cars where the price < 10000.

# In[8]:


#creating new dataset variable name for the problem statements as new_df
new_df= df[df['price'] >= 10000]


# In[9]:


print('Before removing the values less than 10000, the rows & columns are:', df.shape)
print('After removing all the values which are less than 10000, the rows & columns are:', new_df.shape)


# # 2. Which car was bought maximum number of times by customers and whats the highest price for a car in the dataset?

# In[10]:


#checking for the most repitative car name in the dataset
new_df.max().head()


# In[11]:


#printing the highest car price
print('The car that was bought maximum number of times by customers: "vw dasher"')
print('The highest price for a car in the dataset is:', new_df['price'].max())


# # 3. Sort and Return the resultant dataframe in acending order by price.

# In[12]:


new_df.sort_values(by='price', ascending=True)


# # 4. Does Fuel Type affecting the price of cars differently?

# In[13]:


#defining these columns into variables for plotting
fuel_type = new_df['fueltype']
Price = new_df['price']


# In[14]:


from matplotlib import style


# In[15]:


style.use('ggplot')
fig = plt.figure(figsize=(15,5))
fig.suptitle('Fuel type affecting the Price')
plt.subplot()
plt.bar(fuel_type, Price, color='royalblue')
plt.xlabel("Fuel Type")
plt.ylabel("Price")
plt.show()


# Yes, the Fuel Type is do affecting the Price of the cars.
# As shown in the graph above, Diesel cars cost less than the Gas ones.

# # 5. Which variables are significant in predicting the price of a car.

# In[16]:


#defining these columns into variables for plotting
Car_body = new_df['carbody']
door_number = new_df['doornumber']
Aspiration = new_df['aspiration']
Engine_location = new_df['enginelocation']
Drive_wheel = new_df['drivewheel']
Symboling = new_df['symboling']
Fuel_system = new_df['fuelsystem']


# In[17]:


style.use('ggplot')
fig = plt.figure(figsize=(20,15))
fig.suptitle('Variables significant in predicting the price of a car')
plt.subplot(4,2,1)
plt.bar(Aspiration, Price, color='royalblue')
plt.xlabel("Aspiration")
plt.ylabel("Price")
plt.subplot(4,2,2)
plt.bar(Car_body, Price, color='royalblue')
plt.xlabel("Car Body")
plt.ylabel("Price")
plt.subplot(4,2,3)
plt.bar(door_number, Price, color='royalblue')
plt.xlabel("Door Number")
plt.ylabel("Price")
plt.subplot(4,2,4)
plt.bar(fuel_type, Price, color='royalblue')
plt.xlabel("Fuel Type")
plt.ylabel("Price")
plt.subplot(4,2,5)
plt.bar(Symboling, Price, color='royalblue')
plt.xlabel("Symboling")
plt.ylabel("Price")
plt.subplot(4,2,6)
plt.bar(Engine_location, Price, color='royalblue')
plt.xlabel("Engine Location")
plt.ylabel("Price")
plt.subplot(4,2,7)
plt.bar(Drive_wheel, Price, color='royalblue')
plt.xlabel("Drive Wheel")
plt.ylabel("Price")
plt.subplot(4,2,8)
plt.bar(Fuel_system, Price, color='royalblue')
plt.xlabel("Fuel System")
plt.ylabel("Price")
plt.show()


# The variables which are significant in predicting the price of the car are: Aspiration, 
#                                                                             Car Body, 
#                                                                             Door Number, 
#                                                                             Fuel Type, 
#                                                                             Symboling, 
#                                                                             Engine Location, 
#                                                                             Drive wheel, 
#                                                                             fuel System.

# # 6. How well those variables describe the price of a car.

# The variables: 1) Aspiration 2) Fuel Type 3) Drive Wheel 4) Fuel system, describes the price of a car significantly.
# 
# The variables: 1) Door Number 2) Engine Location, doesn't shows any significant marks but as it shows the cars having two doors are most probably sports cars which cost way more than a seadan or four door family cars.
# 
# The variables: 1) Car Body 2) Symboling, has alot of values which makes it difficult to determine which factor causes significant changes in the price of the cars. But still as you can see in the above graphs of carbody and symboling, The HardTop carbody costs more following up with seadan and wagon is comparetively less in cost than the other car body. same in symboling, 1 costs more following up with 0, 2 and -2 costs about the same price.

# # EDA

# Checking for outliers

# In[18]:


#checking some more information
df.describe()


# In[19]:


#detecting outliers
sns.boxplot(df['price'])


# Locating and finding the outliers in teh dataset

# In[20]:


#finding outliers by Z_score method
upper_limit= df['price'].mean() + 3*df['price'].std()
lower_limit= df['price'].mean() - 3*df['price'].std()
print('Upper Limit:', upper_limit)
print('Lower Limit:', lower_limit)


# In[21]:


#all the outliers
df.loc[(df['price'] > upper_limit) | (df['price'] < lower_limit)]


# In[22]:


#trimmimg the outliers(instead of droping the values, created a new dataset as new_df1 which doesn't include the outliers)
#creating new dataset variable name for the further process as new_df1
new_df1= df.loc[(df['price'] < upper_limit) & (df['price'] > lower_limit)]
print('Before removing outliers:', len(df))
print('After removing outliers:', len(new_df))
print('Outliers:', len(df) - len(new_df))


# In[23]:


#printing the new dataset(first 20 rows) without the outliers
new_df1.head(20)


# Correlation between columns

# In[24]:


plt.figure(figsize=(10,7))
sns.heatmap(new_df1.corr(), annot=True)
plt.title('Correlation between the Columns')
plt.show()


# Assinging some column values to 0 and 1's beacause Machine Learning model cannot process string values.

# In[25]:


#checking new_df1 for analyising which column values to renamed
new_df1.head(60)


# Assigning values to these particular columns

# In[26]:


#only those columns which contains 2 or minimunm number of strings in them
#fueltype
new_df1.loc[new_df1['fueltype'] == 'gas', 'fueltype',] = 1
new_df1.loc[new_df1['fueltype'] == 'diesel', 'fueltype',] = 0

#aspiration
new_df1.loc[new_df1['aspiration'] == 'std', 'aspiration',] = 1
new_df1.loc[new_df1['aspiration'] == 'turbo', 'aspiration',] = 0

#doornumber
new_df1.loc[new_df1['doornumber'] == 'two', 'doornumber',] = 1
new_df1.loc[new_df1['doornumber'] == 'four', 'doornumber',] = 0

#drivewheel
new_df1.loc[new_df1['drivewheel'] == 'rwd', 'drivewheel',] = 1
new_df1.loc[new_df1['drivewheel'] == 'fwd', 'drivewheel',] = 2
new_df1.loc[new_df1['drivewheel'] == '4wd', 'drivewheel',] = 0

#enginelocation
new_df1.loc[new_df1['enginelocation'] == 'front', 'enginelocation',] = 1
new_df1.loc[new_df1['enginelocation'] == 'rear', 'enginelocation',] = 0


# In[27]:


#printing new_df1 head to check
new_df1.head(60)


# Checking values of these columns to see how many unique number of strings they contain 

# In[28]:


#enginetype
new_df1['enginetype'].unique()


# In[29]:


#fuelsystem
new_df1['fuelsystem'].unique()


# In[30]:


#carbody
new_df1['carbody'].unique()


# In[31]:


#cylindernumber
new_df1['cylindernumber'].unique()


# 'enginetype','fuelsystem','cylindernumber','carbody' = these four columns contains too many different types of strings which is difficult to assign values to, so dropping these columns for further procedure

# In[32]:


#dropping the columns 'enginetype','fuelsystem','cylindernumber','carbody'
#creating new dataset variable name for further process as new_df2
new_df2= new_df1.drop(['enginetype','fuelsystem','cylindernumber','carbody'], axis=1)


# In[33]:


#checking the head of new_df2 for seeing the columns have droped or not
new_df2.head()


# # Model Building

# In[34]:


#creating X and y matrix
#dropping the Carname because too many strings in the column and price because it needs to be feed in Y
X = new_df2.drop(['CarName','price'], axis=1)
y = new_df2["price"]


# In[35]:


X.shape


# In[36]:


y.shape


# In[37]:


#splitting X and y into training set and testing set
from sklearn.model_selection import train_test_split


# In[38]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=43)


# In[39]:


#normalizing data in columns
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[40]:


#importing Linear Regression
from sklearn.linear_model import LinearRegression


# In[41]:


#instantiating the model
linreg = LinearRegression()


# In[42]:


#fitting the model
linreg.fit(X_train,y_train)


# # Prediction

# In[43]:


y_pred = linreg.predict(X_test)


# In[44]:


#performance evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[45]:


#now printing the MAE, MSE and R2 score
print("MAE: ", (mean_absolute_error(y_pred, y_test)))
print("MSE: ", (mean_squared_error(y_pred, y_test)))
print("R2 score: ", (r2_score(y_pred, y_test)))


# # Regression Plot

# In[46]:


#predicted values vs actual values
sns.regplot(x=y_pred, y=y_test)
plt.xlabel("Predicted Price")
plt.ylabel('Actual Price')
plt.title("Actual vs Predicted Price")
plt.show()


# Final Conclusion: 
# 
# 1) Data collecting: Gathered all the relevent information about the data which are essential for the analysis.
# 2) Data Sanitization: Then performed data sanitization(checking null values and duplicated values), luckily there were none present in dataset.
# 3) Problem Statements: Solved the problem statements first because EDA would messed up my further analysis for ML purposes as outliers were needed to be detected for feeding the model.
# 4) EDA: Outliers were detected, there were 3 outliers present in the dataset, index no.: 16, 73, 74. after that instead of dropping these values from the original dataset, created a new dataset which doesn't include these values as new_df1. After that plotted a heat map to find the better correlation between the variables, then converted those coulmn values whose values where into strings and converted them into 0's and 1's because ML models cannot read string values. These values were removed('enginetype','fuelsystem','cylindernumber','carbody') because these four columns contains too many different types of strings which is difficult to assign values to, so dropping these columns for further procedure would be better option.
# 5) Model Building: Dropped 'Carname' and taken the 'price' as Y, trained the dataset in 70:30 ratio. then imported the linear regression and fitted in the model.  
# 6) Prediction: MeanAbsoluteError: 2224.0553273384357, MeanSquaredError: 9088371.645255297 and R2 score: 0.686423340300427
# 7) Regression Plot(Actual vs Predicted Values): The predicted prices as as close as the actual price till 15000rs but after 15000rs there is difference being observed. most probably because of the 'cylindernumber' column, but as it is a self analysis(done for personal use, mostly skilled based) so there always a room for improvemnets.
