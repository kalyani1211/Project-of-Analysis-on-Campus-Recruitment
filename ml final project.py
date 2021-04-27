#!/usr/bin/env python
# coding: utf-8

# # 1. Imports

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix


# # 2. Loading data

# In[4]:


students = pd.read_csv('C:/Users/Kalyani/Downloads/StudentsPerformance.csv')
students


# For this task, we only need the students' genders and their math, reading and writing scores.

# In[5]:


scores = students[['gender', 'math score', 'reading score', 'writing score']]
scores


# In[7]:


students.describe()


# You can see the descriptive statistics of numerical variables such as total count, mean, standard deviation, 
# minimum and maximum values and three quantiles of the data (25%,50%,75%).

# In[9]:


students.shape


# It shows the number of rows and columns.

# In[10]:


students.isnull().sum() #checks if there are any missing values


# # 3. Checking gender frequencies

# In classification problems it is important to ensure that the target data isn't significantly unbalanced. 
# Therefore for this dataset we need to check the male and female frequencies.

# In[13]:


sns.set_style("darkgrid")
sns.countplot(scores['gender'], palette="Set1")
students['gender'].value_counts()


# Only slightly unbalanced - won't affect the results.
# 
# Now let's visualise the distributions of scores for each gender.
# 
# 

# # 4.Visualising gender scores

# In[14]:


sns.displot(scores, x='math score', hue='gender', palette="Set1")


# Males seem to get slightly higher maths scores than females, but there isn't a massive difference.

# In[16]:


sns.displot(scores, x='reading score', hue='gender', palette="Set1")


# The difference here is more obvious, with female reading scores being much higher than males overall.

# In[17]:


sns.displot(scores, x='writing score', hue='gender', palette="Set1")


# This distribution is similar to reading, with the female writing scores being much higher than males overall.

# In[18]:


x = scores[['math score','reading score', 'writing score']]
y = scores['gender']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)


# # 5. Logistic Regression

# In[19]:


linreg = LogisticRegression()

linreg.fit(x_train, y_train)
print('Logistic Regression Accuracy:', linreg.score(x_test, y_test)*100, '%')


# In[20]:


sns.set_style("white")
disp = plot_confusion_matrix(linreg, x_test, y_test, cmap=plt.cm.PuBu)
disp.ax_.set_title('Confusion Matrix for Logistic Regression')


# # 6. K-Nearest Neighbours

# In[21]:


knn = KNeighborsClassifier()

knn.fit(x_train, y_train)
print('KNN Accuracy:', knn.score(x_test, y_test)*100, '%')


# In[22]:


sns.set_style("white")
disp2 = plot_confusion_matrix(knn, x_test, y_test, cmap=plt.cm.PuBu)
disp2.ax_.set_title('Confusion Matrix for KNN')


# # 7. Gaussian Naive Bayes

# In[23]:


nb = GaussianNB()

nb.fit(x_train, y_train)
print('Gaussian Naive Bayes Accuracy:', nb.score(x_test, y_test)*100, '%')


# In[24]:


sns.set_style("white")
disp3 = plot_confusion_matrix(nb, x_test, y_test, cmap=plt.cm.PuBu)
disp3.ax_.set_title('Confusion Matrix for Gaussian Naive Bayes')


# # 8. Decision Tree Classifier

# In[26]:


dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
print('Decision Tree Accuracy:', dtc.score(x_test, y_test)*100, '%')


# In[27]:


sns.set_style("white")
disp4 = plot_confusion_matrix(dtc, x_test, y_test, cmap=plt.cm.PuBu)
disp4.ax_.set_title('Confusion Matrix for Decision Tree Classifier')


# # 9.SVC (Support Vector Classification)

# In[28]:


svc = SVC()
svc.fit(x_train, y_train)
print('SVC Accuracy:', svc.score(x_test, y_test)*100, '%')


# In[29]:


sns.set_style("white")
disp5 = plot_confusion_matrix(svc, x_test, y_test, cmap=plt.cm.PuBu)
disp5.ax_.set_title('Confusion Matrix for SVC')


# # 10. Conclusion

# it seems that males are better at maths whereas females are better at reading and writing.

# In[ ]:




