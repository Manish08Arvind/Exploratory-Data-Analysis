#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis  - House Price Prediction

# In[1]:


# Import the required libraries/packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Display all columns
pd.set_option('display.max_columns',None)
# Display all rows
# pd.set_option('display.max_rows',None)


# In[2]:


# Data Import

df_hp = pd.read_csv('House_Price_Train.csv')
df_hp.head()


# In[3]:


# Shape of the dataset i.e. rows and columns
df_hp.shape


# In[4]:


# Top 5 Records
df_hp.head(5)


# In[5]:


# Bottom 5 Records
df_hp.tail(5)


# ### Missing Values

# Identifying missing values in the dataset

# In[6]:


# Total Missing values count
df_hp.isnull().sum().sum()


# In[7]:


# Missing values as a percent of Total number of records
print(str(round((df_hp.isnull().sum().sum()/df_hp.size)*100))+'% of total records in the dataset are missing')


# In[8]:


# Count of Missing values for each column
df_hp.isnull().sum()


# In[9]:


# % of Missing values against total records in a column
print(round((df_hp.isnull().sum()/df_hp.shape[0])*100,0))


# In[10]:


# Number of columns that have missing values along with their count

missing_features_list = [i for i in list(df_hp.columns) if df_hp[i].isnull().sum()>0]
missing_features_list


# In[11]:


for j in missing_features_list:
    print(j,df_hp[j].isnull().sum())


# In[12]:


for j in missing_features_list:
    print(j,str(round((df_hp[j].isnull().sum()/df_hp.isnull().sum().sum())*100,0))+'%')


# In[13]:


# Finding relationship between Missing Values and Sales Price
    
for i in missing_features_list:
    df = df_hp.copy()
    
    # A variable that indicates 1 if the observation was missing or 0 if values is non-null
    df[i] = np.where(df[i].isnull(),1,0)
    
    # Taking Median Sale Price into consideration
    df.groupby(i)['SalePrice'].median().plot.bar(color=['blue','orange'])
    plt.title(i)
    plt.show()


# The above plots depict that most of the features with NaN values have a higher median price than with their non-null values. So, this needs to be handled in the feature engineering.

# In[14]:


# Feature - Id is unique and not required for further analysis
print('Id of houses {}'.format(len(df_hp['Id'])))


# ### Numerical Variables

# In[15]:


numerical_features = [i for i in df_hp.columns if df_hp[i].dtype!='O']
numerical_features


# In[16]:


print('Number of numerical variables are {}'.format(len(numerical_features)))


# ### Temporal Variables (Example: Datetime variables)

# In[17]:


# List of variables that contains year 

year_features = [j for j in numerical_features if 'Yr' in j or 'Year' in j]
year_features


# In[18]:


for k in year_features:
    print(k,df_hp[k].unique())


# In[19]:


df_hp.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('Median House Price')
plt.title('House Price vs YearSold')


# In[20]:


# Comparing House Prices with difference between year sold and other year variables

for i in year_features:
    if i!='YrSold':
        df = df_hp.copy()
        
        # Difference between Year variable and Year Sold
        df[i] = df['YrSold']-df[i]
        
        plt.scatter(df[i],df['SalePrice'])
        plt.xlabel(i)
        plt.ylabel('SalePrice')
        plt.show()


# Numerical variables are of two types: Discrete and Continuous

# ### Discrete Features

# In[21]:


discrete_features = [i for i in numerical_features if df_hp[i].nunique()<=25 and i not in year_features+['Id']]
discrete_features


# In[22]:


print('Discrete variable count',len(discrete_features))


# In[23]:


# Plotting relationship between Sale Price and Discrete variables

for i in discrete_features:
    df_hp.groupby(i)['SalePrice'].median().plot.bar()
    plt.xlabel(i)
    plt.ylabel('Median Sale Price')
    plt.title(str(i)+' vs Sale Price')
    plt.show()


# In[24]:


# Overall Quality and Sale Price has a monotonic relationship


# ### Continuous Features

# In[25]:


continuous_features = [i for i in numerical_features if i not in discrete_features+year_features+['Id']]
continuous_features


# In[26]:


# Histograms to plot continuous variables

for i in continuous_features:
    df = df_hp.copy()
    df[i].hist(bins=25)
    plt.xlabel(i)
    plt.ylabel('Count')
    plt.title(i)
    plt.show()


# In[27]:


# Log Transformation for continuous features

for i in continuous_features:
    df = df_hp.copy()
    if 0 not in df[i].unique(): # This is because log(0) is undefined
        df[i] = np.log(df[i])
        df['SalePrice']=np.log(df['SalePrice'])
        plt.scatter(x=df[i],y=df['SalePrice'])
        plt.xlabel(i)
        plt.ylabel('SalePrice')
        plt.title(str(i)+' vs SalePrice')
        plt.show()


# ### Outliers

# In[28]:


for i in continuous_features:
    df = df_hp.copy()
    if 0 not in df[i].unique(): # This is because log(0) is undefined
        df[i] = np.log(df[i])
        df.boxplot(column=i)
        plt.ylabel(i)
        plt.title(i)
        plt.show()


# ### Categorical Features

# In[29]:


catg_features = [i for i in list(df_hp.columns) if df_hp[i].dtype=='O']
catg_features


# In[30]:


df_hp[catg_features].head()


# In[31]:


for i in catg_features:
    print('Feature: {} - Number of categories are {}'.format(i,df_hp[i].nunique()))


# In[32]:


# Relationship between Categorical variables and dependent variable


# In[33]:


for i in catg_features:
    df = df_hp.copy()
    grouped_df = pd.DataFrame(df.groupby(i)['SalePrice'].median(),columns=['SalePrice']).reset_index()
    sns.barplot(data=grouped_df,x=i,y='SalePrice',hue=i)
    plt.xlabel(i)
    plt.ylabel('Sale Price')
    plt.show()

