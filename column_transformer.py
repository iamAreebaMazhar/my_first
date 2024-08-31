#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd 


# In[5]:


df=pd.read_csv('covid_toy.csv')


# In[7]:


df.head(5)


# In[9]:


df['gender'].value_counts()


# In[10]:


df['cough'].value_counts()


# In[13]:


df['city'].value_counts()


# In[15]:


df.isnull().sum()


# In[16]:


df.duplicated().sum()


# In[20]:


df.drop_duplicates(inplace=True)


# In[21]:


df.shape


# # Simple way of doing Encoding

# In[33]:


from sklearn.model_selection import train_test_split
#So, df.drop('has_covid', axis=1) means that you are dropping the column named 'has_covid'
#from the DataFrame df 
 #axis=1 refers to column name which you want to drop
X_train,X_test,y_train,y_test=train_test_split(df.drop('has_covid',axis=1),
                                              df['has_covid'],
                                              test_size=0.2)


# In[34]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder


# In[35]:


si=SimpleImputer()
X_train_fever=si.fit_transform(X_train[['fever']])
X_test_fever=si.fit_transform(X_test[['fever']])


# In[36]:


# X_train_fever


# In[39]:


X_train_fever.shape


# In[40]:


df.head(2)


# In[41]:


ohe=OneHotEncoder(drop='first',sparse=False)
X_train_gender_city=ohe.fit_transform(X_train[['gender','city']])
X_test_gender_city=ohe.fit_transform(X_test[['gender','city']])


# In[44]:


X_train_gender_city.shape


# In[46]:


oe=OrdinalEncoder(categories=[['Mild','Strong']])
X_train_cough=oe.fit_transform(X_train[['cough']])
X_test_cough=oe.fit_transform(X_test[['cough']])


# In[50]:


X_train_cough.shape


# In[52]:


# concatenate all the columns but first extract age columns from X_train
X_train_age=X_train.drop(columns=['gender','fever','cough','city'])
X_test_age=X_test.drop(columns=['gender','fever','cough','city'])


# In[54]:


X_train_age.shape


# In[56]:


# now concatenate all the columns
X_train_transform=np.concatenate((X_train_age,X_train_fever,X_train_gender_city,X_train_cough),axis=1) 
X_test_transform=np.concatenate((X_test_age,X_test_fever,X_test_gender_city,X_test_cough),axis=1)   


# In[59]:


X_train_transform.shape


# # Column Tranformer

# In[61]:


from sklearn.compose import ColumnTransformer


# In[70]:


transformer=ColumnTransformer(transformers=[
    ('tnf1',SimpleImputer(),['fever']),
    ('tnf2',OneHotEncoder(sparse=False,drop='first'),['gender','city']),
    ('tnf3',OrdinalEncoder(categories=[['Mild','Strong']]),['cough'])
    
],remainder='passthrough')


# In[72]:


transformer.fit_transform(X_train).shape


# In[74]:


transformer.fit_transform(X_test).shape


# In[ ]:




