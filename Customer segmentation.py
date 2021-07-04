#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn import set_config
set_config(print_changed_only=False)


# In[2]:


data=pd.read_csv('Supermarket_CustomerMembers.csv')


# In[3]:


df=data
df.head()


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


df=df.drop(['CustomerID'],axis=1)


# In[7]:


df.head()


# In[8]:


df=df.rename(columns={'Annual Income (k$)':'Income','Spending Score (1-100)':'Spending_Score'})
df.head()


# In[9]:


sns.scatterplot(df['Income'],df['Spending_Score'],data=df)


# In[10]:


x=df[['Income','Spending_Score']]


# In[11]:


x.head()


# In[12]:


x.shape


# In[13]:


from sklearn.cluster import KMeans


# In[14]:


from sklearn.preprocessing import StandardScaler


# In[15]:


scaler=StandardScaler()


# In[16]:


X_scaled=scaler.fit_transform(df[['Income','Spending_Score']])


# In[17]:


X_Scaled=pd.DataFrame(data=X_scaled,columns=[['Income','Spending_Score']])


# In[18]:


X_Scaled.head()


# In[19]:


wss=[]

for i in range(1,10):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=50)
    kmeans.fit(X_Scaled)
    wss.append(kmeans.inertia_)


# In[20]:


plt.plot(range(1,10),wss)
plt.title('Elbow graph')
plt.xlabel('Number of K(clusters)')
plt.ylabel('WCSS')
plt.show()


# In[21]:


kmeans=KMeans(n_clusters=5,init='k-means++')


# In[22]:


Y=kmeans.fit_predict(X_Scaled)


# In[33]:


clusters=kmeans.cluster_centers_


# In[24]:


kmeans.labels_


# In[25]:


df['Label']=kmeans.labels_


# In[26]:


df.head()


# In[27]:


df['Label'].value_counts()


# In[35]:



plt.figure(figsize=(15,10))
sns.scatterplot(df['Income'],df['Spending_Score'],hue='Label',palette='tab10',data=df)


# In[29]:


## Customer Profiling

## we see that the customers from label 0 are our target customers as they earn more and spend more. They are target customers

## the customers from label 2 are the 'Selective buyers'. They earn a lot but are not easily willing to spend. We can give them
# royalties and do effective marketing to turn them into target customers

## the customers from label 1 are the 'Potential targets'. They can be chnaged into our targeted customers with special offers.

## customers from label 4 are 'Spending class'. They don't earn much but spend more.

## customers from label 3 are 'Wise Shoppers', they earn less so spend less. They can be given discounts to move them to 
    #spending class group


# In[30]:


df['Category']=df.Label.map({0:'Target Customers',1:'Potential Targets',2:'Selective buyers',3:'Wise Shoppers',4:'Spending Class'})


# In[31]:


df.head()


# In[32]:


df


# In[ ]:




