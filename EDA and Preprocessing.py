#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### EDA and preprocessing

# ##### Explore the data, list down the unique values in each feature and find its length.Perform the statistical analysis and renaming of the columns
# 

# In[3]:


data = pd.read_csv(r"C:\Users\user\Downloads\Employee.csv")
data


# In[4]:


data.info()


# ####  list down the unique values in each feature and find its length. 

# In[4]:


#creating a dictionary to store the unique values for each feature:
unique_values = {col: data[col].unique() for col in data.columns}  
#finding length of each feature:
unique_length = {col: len(data[col].unique()) for col in data.columns}

for col in unique_values:
    print("Feature: ", col)
    print("Unique values: ",unique_values[col])
    print("Length: ",unique_length[col])
    print("-" * 100)
    


# In[11]:


data.describe()


# ##### Perform the statistical analysis and renaming of the columns.

# In[5]:


#REnaming columns:
data.rename(columns = {"Age":"Employee_Age","Salary":"Employee_Salary",},inplace = True)
#Renaming company names:
data['Company'] = data['Company'].replace('Infosys Pvt Lmt','Infosys')
data['Company'] = data['Company'].replace('Tata Consultancy Services','TCS')
data['Company'] = data['Company'].replace('Cognizant','CTS')
#Lets display the data after renaming:
data.describe()


# In[6]:


#Printing unique values after renaming:
unique_values = {col: data[col].unique() for col in data.columns}  
for col in unique_values:
    print("Feature: ", col)
    print("Unique values: ",unique_values[col])
    print("-" * 100)


# In[7]:


# displaying data after name correction:
data


# ### Data Cleaning  

# Find the missing and inappropriate values, treat them appropriately.  
# Remove all duplicate rows.  
# Find the outliers.  
# Replace the value 0 in age as NaN  
# Treat the null values in all columns using any measures(removing/ replace the values with mean/median/mode)

# In[ ]:





# In[8]:


#finding missing values:
missing_values = data.isnull().sum()
missing_values


# In[9]:


#Checking mean of the null values:
round(data.isnull().mean()*100,2)


# In[10]:


#Replacing missing values

#Replacing 0 in 'Employee_Age' as NaN
data['Employee_Age'].replace(0,np.nan,inplace=True)
#Replacing missing 'Employee_Age' using median
data['Employee_Age'].fillna(data['Employee_Age'].median(), inplace=True)
#Replacing missing 'Employee_Salary' using median
data['Employee_Salary'].fillna(data["Employee_Salary"].median(), inplace=True)
#Replacing missing 'Company' using mode
data['Company'].fillna(data['Company'].mode()[0], inplace=True)
#Replacing missing 'Place' using mode
data['Place'].fillna(data['Place'].mode()[0], inplace=True)
data.isnull().sum()


# In[11]:


#Checking for duplicates
data.duplicated().sum()


# In[12]:


#Removing the duplicates
data = data.drop_duplicates()
data.info()


# In[13]:


# finding outliers
#Checking skewness
data.skew()


# In[14]:


#Let's find and remove outliers from 'Employee_Age' column
Q1 = data.Employee_Age.quantile(0.25)
print("Q1: ",Q1)
Q3 = data.Employee_Age.quantile(0.75)
print("Q3: ",Q3)
IQR = Q3 - Q1
print("IQR: ",IQR)


# In[16]:


lower_bound = Q1-1.5*IQR
print("Lower Bound: ",lower_bound)
upper_bound = Q3+1.5*IQR
print("Upper Bound: ",upper_bound)
#finding outliers
Outliers=data[(data.Employee_Age<lower_bound) | (data.Employee_Age>upper_bound)]
data
#Removing outliers
data_cleaned=data[(data.Employee_Age>=lower_bound) & (data.Employee_Age<=upper_bound)]
data_cleaned


# In[17]:


#plotting the result
plt.subplot(1,2,1)
sns.boxplot(x = data['Employee_Age'])
plt.title("Original data")
plt.subplot(1,2,2)
sns.boxplot(x = data_cleaned['Employee_Age'])
plt.title("Data After Removing outliers")
plt.show()


# In[18]:


#Removing outliers from 'Employee_Salary' column

Q1 = data.Employee_Salary.quantile(0.25)
print("Q1: ",Q1)
Q3 = data.Employee_Salary.quantile(0.75)
print("Q3: ",Q3)
IQR = Q3 - Q1
print("IQR: ",IQR)


# In[19]:


lower_bound = Q1-1.5*IQR
print("Lower Bound: ",lower_bound)
upper_bound = Q3+1.5*IQR
print("Upper Bound: ",upper_bound)
#finding outliers
Outliers=data[(data.Employee_Salary<lower_bound) | (data.Employee_Salary>upper_bound)]
data
#Removing outliers
data_cleaned=data[(data.Employee_Salary>=lower_bound) & (data.Employee_Salary<=upper_bound)]
data_cleaned


# In[20]:


#plotting the result
plt.subplot(1,2,1)
sns.boxplot(x = data['Employee_Salary'])
plt.title("Original data")
plt.subplot(1,2,2)
sns.boxplot(x = data_cleaned['Employee_Salary'])
plt.title("Data After Removing outliers")
plt.show()


# ### Data Analysis

# Filter the data with age >40 and salary<5000  
#  Plot the chart with age and salary  
#  Count the number of people from each place and represent it visually 

# In[21]:


#filtering the data with age>40 and salary<5000
filter = (data['Employee_Age']>40) & (data['Employee_Salary']<5000)
filtered_data = data[filter]
filtered_data


# In[22]:


#Plotting chart with age and salary
plt.subplot(1,2,1)
plt.scatter(data_cleaned['Employee_Age'],data_cleaned['Employee_Salary'])
plt.xlabel("Employee Age")
plt.ylabel("Employee Salary")
plt.title("Age vs Salary Original")
plt.subplot(1,2,2)
plt.scatter(filtered_data['Employee_Age'],filtered_data['Employee_Salary'])
plt.xlabel("Employee Age")
plt.ylabel("Employee Salary")
plt.title("Age vs Salary Filtered")
plt.tight_layout()
plt.show()


# In[23]:


# counting the no.of people from each place and represent it visually 
count = data['Place'].value_counts()
print("No.of people from each place: ",count)


# In[24]:


place_count = count.index
employees = count.values

#Visualizing the counts
plt.figure(figsize=[10,6])
plt.bar(place_count,employees)
plt.xlabel("Place")
plt.ylabel("No.of people")
plt.title("No.of people from each place")
plt.tight_layout()
plt.show()


# ### Data Encoding:

# Convert categorical variables into numerical representations using techniques such as one-hot encoding, 
# label encoding, making them suitable for analysis by machine learning algorithms. 
# 

# In[27]:


from sklearn.preprocessing import OneHotEncoder
#one-hot encoding
one_hot_encoder = OneHotEncoder(sparse=False)
encoded_columns = one_hot_encoder.fit_transform(data[['Company', 'Place']])
encoded_features = one_hot_encoder.get_feature_names_out(['Company', 'Place'])
one_hot_encoded_data = pd.DataFrame(encoded_columns, columns=encoded_features)
one_hot_encoded_data


# In[30]:



#Label encoding
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoded_data = data.copy() # make a copy or original dataframe
colums_to_encode = ['Company','Place','Country']
for column in colums_to_encode:
    label_encoded_data[column] = label_encoder.fit_transform(label_encoded_data[column])
label_encoded_data


# ### Feature Scaling:

#  After the process of encoding, perform the scaling of the features using standardscaler and minmaxscaler.

# In[33]:


# perform the scaling of the features using standard scaler (StandardScaler applied to label encoded  data)
from sklearn.preprocessing import StandardScaler
sd_scaler = StandardScaler()
new_data = label_encoded_data 
new_data[['Employee_Age','Employee_Salary']]=sd_scaler.fit_transform(new_data[['Employee_Age','Employee_Salary']])
new_data
     


# In[39]:


#Scaling using minmaxscaler 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
scaled_data = scaler.fit_transform(one_hot_encoded_data)
scaled_df = pd.DataFrame(scaled_data,columns=one_hot_encoded_data.columns)
scaled_df  

