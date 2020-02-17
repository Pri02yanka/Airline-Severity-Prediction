#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# Plotting graphs
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 


# In[2]:


train_data= pd.read_csv('C:/Users/priyanka.CHEF5GWTN2/Desktop/Hackathon/Airplane Accident/3c055e822d5b11ea/train.csv', encoding='utf-8-sig')
validate_data= pd.read_csv('C:/Users/priyanka.CHEF5GWTN2/Desktop/Hackathon/Airplane Accident/3c055e822d5b11ea/test.csv', encoding='utf-8-sig')


# In[3]:


train_data.columns


# In[4]:


validate_data.columns


# In[5]:


train_data.dtypes


# Categorical features: Accident_Type_Code, Violations
# Numerical features: Safety_Score, Days_Since_Inspection, Total_Safety_Complaints, Control_Metric,Turbulence_In_gforces, Cabin_Temperature, Max_Elevation, Adverse_Weather_Metric

# In[6]:


train_data.head()


# In[7]:


# Missing values in train data
pd.DataFrame({'missing_counts': train_data.apply(lambda x: np.sum(x.isnull())), 'unique': train_data.apply(lambda x: x.nunique()), 'data_types': train_data.dtypes})


# In[8]:


train_data.Severity.value_counts()


# In[9]:


# Univariate analysis with Histogram
cols = ['Safety_Score', 'Days_Since_Inspection', 'Total_Safety_Complaints', 'Control_Metric', 'Turbulence_In_gforces','Cabin_Temperature','Max_Elevation', 'Adverse_Weather_Metric']
for col in cols: 
    plt.figure(1)
    sns.distplot(train_data[col])
    plt.show()


# In[10]:


# Bivariate Analysis- Nominal & Ordinal Variables
plt.figure(1)
plt.subplot(121)
train_data.Accident_Type_Code.value_counts().plot.bar(figsize=(20,5))
plt.title('Accident_Type_Code', fontsize=10)
plt.xlabel('Accident_Type_Code', fontsize = 20.0)
plt.ylabel('Count', fontsize = 20.0)

plt.subplot(122)
train_data.Violations.value_counts().plot.bar(figsize=(20,5))
plt.title('Violations', fontsize=10)
plt.xlabel('Violations', fontsize = 20.0)
plt.ylabel('Count', fontsize = 20.0)


# In[11]:


# Split train, test and validate data

from sklearn.model_selection import train_test_split
X = train_data.drop(['Accident_ID','Severity'],axis=1)
y = train_data['Severity']

validate_X = validate_data.drop(['Accident_ID'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)


# In[12]:


# Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('Train data Logistic regression Accuracy: {:.3f}'.format(accuracy_score(y_train, logreg.predict(X_train))))
print('Test data Logistic regression accuracy: {:.3f}'.format(accuracy_score(y_test, logreg.predict(X_test))))


# In[13]:


# Decision Tree Model

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import GridSearchCV

param_grid = {'max_depth': np.arange(3, 10)}

tree = GridSearchCV(DecisionTreeClassifier( max_depth = 9), param_grid, cv = 10)
tree.fit( X_train, y_train )
print('Train data Decision Tree Accuracy: {:.3f}'.format(accuracy_score(y_train, tree.predict(X_train))))
print('Test data Decision Tree Accuracy: {:.3f}'.format(accuracy_score(y_test, tree.predict(X_test))))


# In[14]:


# Random Forest Model

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
print('Train data Random Forest Accuracy: {:.3f}'.format(accuracy_score(y_train, rf.predict(X_train))))
print('Test data Random Forest Accuracy: {:.3f}'.format(accuracy_score(y_test, rf.predict(X_test))))


# In[15]:


from sklearn.metrics import classification_report
print(classification_report(y_test, rf.predict(X_test)))


# In[21]:


# Confusion matrix of random forest
from sklearn.metrics import confusion_matrix
print (" Confusion matrix ", confusion_matrix(y_test, rf.predict(X_test)))


# In[17]:


# cnf_matrix = confusion_matrix(y_train, y_pred,labels=['Highly_Fatal_And_Damaging', 'Minor_Damage_And_Injuries', 'Significant_Damage_And_Fatalities', 'Significant_Damage_And_Serious_Injuries'])
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=['Highly_Fatal_And_Damaging', 'Minor_Damage_And_Injuries', 'Significant_Damage_And_Fatalities', 'Significant_Damage_And_Serious_Injuries'])


# In[22]:


# Support Vector Machine

from sklearn.svm import SVC
svc = SVC(probability=True)
svc.fit(X_train, y_train)
prediction=svc.predict(X_test)
print('Support vector machine accuracy: {:.3f}'.format(accuracy_score(y_train, svc.predict(X_train))))
print('Support vector machine accuracy: {:.3f}'.format(accuracy_score(y_test, prediction)))


# In[23]:


importances=pd.Series(rf.feature_importances_, index=X.columns).sort_values()
importances.plot(kind='barh', figsize=(10,5))
plt.xlabel('Importance of feature - Score')
plt.ylabel('Features')
plt.title("Feature Importance- RandomForest")


# In[20]:


pred_test = pd.DataFrame(rf.predict(validate_X))
pred_test.columns = ['Severity']


# In[24]:


submission_data= validate_data[['Accident_ID']]
submission_data['Severity'] = pred_test['Severity']
submission_data.to_csv('C:/Users/priyanka.CHEF5GWTN2/Desktop/Hackathon/Airplane Accident/3c055e822d5b11ea/submission.csv', index=False)


# In[ ]:




