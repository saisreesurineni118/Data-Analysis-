#!/usr/bin/env python
# coding: utf-8

# ##Introduction:
# ---
# The Titanic dataset is a well-known dataset used for machine learning and statistical analysis. It contains information about
# passengers aboard the RMS Titanic, which tragically sank on April 15, 1912, after hitting an iceberg. The dataset is widely used
# for classification tasks, particularly predicting passenger survival based on different features.

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import math

td = pd.read_csv("train.csv")


# In[123]:


td.head(5)


# In[124]:


td.info()


# In[125]:


td.Survived.value_counts()


# In[126]:



a = td[td["Survived"] == 1]
a1 = a[a["Pclass"] == 1]
print(a1['Age'].median()) 

b = td[td["Survived"] == 0]
b1 = b[b["Pclass"] == 1]
print(b1['Age'].median())

c = td[td["Survived"] == 1]
c1 = c[c["Pclass"] == 2]
print(c1['Age'].median())

d = td[td["Survived"] == 0]
d1 = d[d["Pclass"] == 2]
print(d1['Age'].median())

e = td[td["Survived"] == 1]
e1 = e[e["Pclass"] == 3]
print(e1['Age'].median())

f = td[td["Survived"] == 0]
f1 = f[f["Pclass"] == 3]
print(f1['Age'].median())


a = td[td["Survived"] == 1]
a1 = a[a["Pclass"] == 1]
print(a1['Embarked'].mode()[0])  

b = td[td["Survived"] == 0]
b1 = b[b["Pclass"] == 1]
print(b1['Embarked'].mode()[0])

c = td[td["Survived"] == 1]
c1 = c[c["Pclass"] == 2]
print(c1['Embarked'].mode()[0])

d = td[td["Survived"] == 0]
d1 = d[d["Pclass"] == 2]
print(d1['Embarked'].mode()[0])

e = td[td["Survived"] == 1]
e1 = e[e["Pclass"] == 3]
print(e1['Embarked'].mode()[0])

f = td[td["Survived"] == 0]
f1 = f[f["Pclass"] == 3]
print(f1['Embarked'].mode()[0])


# In[127]:


td.loc[(td["Survived"] == 1) & (td["Pclass"]==1)&(td["Age"].isna()) , "Age"] = td["Age"].fillna(35.0)
td.loc[(td["Survived"] == 1) & (td["Pclass"]==2)&(td["Age"].isna()) , "Age"] = td["Age"].fillna(28.0)
td.loc[(td["Survived"] == 1) & (td["Pclass"]==3)&(td["Age"].isna()) , "Age"] = td["Age"].fillna(22.0)

td.loc[(td["Survived"] == 0) & (td["Pclass"]==1)&(td["Age"].isna()) , "Age"] = td["Age"].fillna(45.25)
td.loc[(td["Survived"] == 0) & (td["Pclass"]==2)&(td["Age"].isna()) , "Age"] = td["Age"].fillna(30.5)
td.loc[(td["Survived"] == 0) & (td["Pclass"]==3)&(td["Age"].isna()) , "Age"] = td["Age"].fillna(25.0)

td.loc[(td["Survived"] == 1) & (td["Pclass"]==1)&(td["Cabin"].isna()) , "Cabin"] = td["Cabin"].fillna("B96")
td.loc[(td["Survived"] == 1) & (td["Pclass"]==2)&(td["Cabin"].isna()) , "Cabin"] = td["Cabin"].fillna("E101")
td.loc[(td["Survived"] == 1) & (td["Pclass"]==3)&(td["Cabin"].isna()) , "Cabin"] = td["Cabin"].fillna("E121")

td.loc[(td["Survived"] == 0) & (td["Pclass"]==1)&(td["Cabin"].isna()) , "Cabin"] = td["Cabin"].fillna("C124")
td.loc[(td["Survived"] == 0) & (td["Pclass"]==2)&(td["Cabin"].isna()) , "Cabin"] = td["Cabin"].fillna("D")
td.loc[(td["Survived"] == 0) & (td["Pclass"]==3)&(td["Cabin"].isna()) , "Cabin"] = td["Cabin"].fillna("F")
td.Embarked.fillna("S")


# In[128]:


# td['FamilySize'] = td['SibSp'] + td['Parch'] + 1


# # 1.Total No.of Passangers:891

# # 2.columns/ filds:12

# In[129]:


sb.countplot(x="Survived",data=td)


# In[130]:


sb.countplot(x="Pclass", data=td)


# In[131]:


sb.countplot(x="Pclass", hue="Sex",data=td)


# In[132]:


sb.countplot(x="Survived", hue="Pclass",data=td)


# In[133]:


sb.countplot(x="Survived", hue="Age",data=td)


# In[134]:


td.isnull()


# In[135]:


td.isnull().sum()


# In[136]:


td.Embarked.value_counts()


# In[137]:


td.dropna(inplace=True)


# In[138]:


td.isnull().sum()


# In[139]:


td.isnull().info()


# In[140]:


print(td['Sex'])


# In[141]:


sex=pd.get_dummies(td['Sex'])


# In[142]:


print(sex)


# In[143]:


print(td['Embarked'])


# In[144]:


emb=pd.get_dummies(td['Embarked'])


# In[145]:


print(emb)


# In[146]:


print(td['Pclass'])


# In[147]:


pcls=pd.get_dummies(td['Pclass'],prefix="Pclass")


# In[148]:


print(pcls)


# In[149]:


print(td.Cabin)


# In[150]:


cab=pd.get_dummies(td['Cabin'],prefix="Cabin")
print(cab)


# In[151]:


td.drop(["Pclass","Sex","Embarked"],axis=1,inplace=True)


# In[152]:


td=pd.concat([td,sex,emb,pcls,cab],axis=1)


# In[153]:


print(td.info())


# In[154]:


td.drop(["Name","PassengerId","Ticket","Cabin"],axis=1,inplace=True)


# In[155]:


td.info()


# In[156]:


td.head(5)


# In[157]:


td['Age_d']=td['Age']
td['SibSp_d']=td['SibSp']
td


# In[180]:


td.corr()


# In[185]:


import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = sns.load_dataset('titanic')  # Correct dataset name 'titanic' (not 'train')

# Compute correlation matrix
correlation = data.corr()

# Create the heatmap
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')

# Set the title
plt.title('Correlation Heatmap')
plt.show()


# In[158]:


X=td.drop(["Survived"],axis=1)


# In[159]:


print(X)


# In[160]:


X.info()


# In[161]:


y=td["Survived"]


# In[162]:


print(y)


# In[163]:


from sklearn.model_selection import train_test_split


# In[164]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=1)


# In[165]:


X_train.info()


# In[166]:


X_test.info()


# In[167]:


from sklearn.linear_model import LogisticRegression


# In[168]:


lm=LogisticRegression(max_iter=10000)


# In[169]:


print(lm.fit(X_train,y_train))


# In[170]:


Predections=lm.predict(X_test)


# In[171]:


from sklearn.metrics import classification_report


# In[172]:


print(classification_report(y_test,Predections))


# In[173]:


from sklearn.metrics import confusion_matrix


# In[174]:


print(confusion_matrix(y_test,Predections))


# In[175]:


from sklearn.metrics import accuracy_score


# In[176]:


print(accuracy_score(y_test,Predections))


# In[177]:


#import pickle


# In[178]:


# save the model to disk
#with open('Titanic','wb') as f:
#   pickle.dump(td,f)


# In[179]:


Predections


# ##Conclusion
# -----
# The Titanic dataset is an excellent resource for practicing data preprocessing, feature engineering, and predictive modeling. 
# It demonstrates the importance of data cleaning and exploratory data analysis (EDA) in real-world datasets. Through this 
# analysis, we can observe social class disparities and survival biases, making it a compelling case study for machine learning 
# applications.
