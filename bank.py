#!/usr/bin/env python
# coding: utf-8

# ## steps to EDA
# step 1:first read the dataset very carefully and understand the dataset
# 
# step2:we have to import the all four libraries
# 
# step3:we have to load the dataset 
# 
# step4:we have to look the column information using he function df.info()
# 
# step4:we have to carefully the highest and lowest values by using the formulas df.describe 
# 
# step5:after that we have to check first 5 rows of our dataset using df.head() and last value df.tail()
# 

# ## 1.BANKING TRANSACTION ANALYSIS(EDA)
# ### A complete exploratort data analysis (EDA)project on large scale banking dataset the project will include
# 1.A realistic dataset 
# 2.step byb step python code with explanations
# 3.visualizations and insights

# ### objective:
# 1.Analyze banking transactions from a large dataset(1M+records).
# 2.Identify trends,customer behavior,and fraud patterns.
# 3.perform exploratory data analysis (EDA) and visualize innsights.
# 
# ### Dataset used:
# Massive Bank Dataset(1M+rows)
# #### Features include:
# 1.TransactionID:unique identifier for each transaction 
# 2.customerId:UNIQUE IDENTIFIER for each cuastomer
# 3.TransactionDate:Timestamp of transaction 
# 4.TransactionType:(credit,debit,transfer,etc)
# 5.amount transaction value
# 6.location :where the transaction happened
# 7.is fraud:1 if fraud,0 if genuine
# 

# ## 2.Import Libraries &Load Data

# In[54]:


# import required libraries
import pandas as pd #for data manupulations 
import numpy as np#for numerical analysis
import matplotlib.pyplot as plt#for visualizations 
import seaborn as sns #satistical analysis


# In[55]:


#Load dataset 
df =pd.read_csv(r"C:\Users\mansw\Downloads\banking_transactions_1M.csv")
#display 5 first records
df.head()


# In[56]:


df.describe()


# In[57]:


df.describe(include="object")


# In[58]:


df.columns


# In[59]:


#missing values
df.isnull().sum()


# ## 5.transaction type analysis

# what are the unique transactions types

# In[60]:


df.info()


# In[61]:


df["TransactionType"].value_counts()


# In[62]:


df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])


# #what is the most common transaction type 

# ##### what are the Location
# df["Location"].value_counts()

#  what are the   TransactionID

# In[63]:


df["TransactionID"].value_counts()


# what are the   CustomerID  
# 
# 

# In[64]:


df["CustomerID"].value_counts()


# ##### what are the amount

# In[65]:


df["Amount"].value_counts()


# #what is the most common transaction types

# In[66]:


sns.countplot(data=df, x="TransactionType")
plt.title("transaction type distribution")#to add the title name of visualization 
plt.show()#to show the visualization 


# In[67]:


sns.countplot(data=df, x="Location")
plt.title("Location")#to add the title name of visualization 
plt.show()


# In[68]:


sns.countplot(data=df, x="isFraud")
plt.title("isFraud")#t add the title name of visualization 
plt.show()


# #what is the average transaction amount

# In[69]:


df.info()
df["Amount"].mean()


# In[70]:


df["Amount"].median()


# In[71]:


df["Amount"].mode()


# #what is the distributon of transaction amounts

# In[72]:


plt.figure(figsize=(10,5))

#plt.figure(figsize=(10, 5)): This line creates a new figure (a blank canvas) for plotting.
#The figsize argument specifies the size of the figure in inches. 
#Here, the width is 10 inches and the height is 5 inches.
#This ensures the plot will be large and easy to view.
sns.histplot(df["Amount"], bins=30,kde=True)
"""sns.histplot(...): This is a function from Seaborn (a visualization library built on top of Matplotlib) to create a histogram plot. The histogram shows the distribution of data, i.e., how data is spread out.
df["Amount"]: This refers to a specific column in your DataFrame df, specifically the "Amount" column. It represents the transaction amounts in the dataset.
bins=50: This defines how many bins (intervals) to use for the histogram. In this case, the data will be divided into 50 intervals to show the distribution.
kde=True: This adds a Kernel Density Estimation (KDE) curve to the histogram. The KDE curve is a smooth line that estimates the probability density function of the data. It helps visualize the underlying distribution of the data in addition to the histogram."""
plt.title("Transaction Amount Distribution")
#plt.title("Transaction Amount Distribution"): This line adds a title to the plot. The title here is "Transaction Amount Distribution," which helps the viewer understand what the plot is showing (distribution of transaction amounts).
plt.xlabel("Amount")
"""plt.xlabel("Amount"): This line sets the label for the x-axis (horizontal axis) to "Amount". This tells the viewer that the x-axis represents transaction amounts."""
plt.ylabel("Frequency")
"""plt.ylabel("Frequency"): This line sets the label for the y-axis (vertical axis) to "Frequency". This tells the viewer that the y-axis represents the frequency (or count) of transactions in each bin."""
plt.show()
"""plt.show(): Finally, this command displays the plot on the screen. It renders the plot so that it can be viewed."""


# ### customer behavior analysis
# #how many unique customer are there 
# 

# In[73]:


df["Location"].nunique()


# #which customer have the highest total transaction

# In[74]:


df.groupby("CustomerID")["Amount"].sum().sort_values(ascending=False).head()


# In[75]:


df.groupby("CustomerID")["Amount"].sum().sort_values(ascending=True).tail()


# In[76]:


df.groupby("CustomerID")["Amount"].sum().sort_values(ascending=False).head(12)


# """df.groupby("CustomerID"):
# 
# groupby("CustomerID"): This groups the data in the DataFrame df by the "CustomerID" column. This means the rows in the DataFrame are grouped based on unique customer IDs. After grouping, you can perform operations on each group (in this case, customer) individually.
# ["Amount"].sum():
# 
# ["Amount"]: After grouping by "CustomerID", we are focusing on the "Amount" column, which presumably contains the transaction amounts.
# .sum(): This aggregates (sums up) the "Amount" values for each customer. It calculates the total spending per customer by summing up all the transaction amounts for each customer ID.
# .sort_values(ascending=False):
# 
# .sort_values(ascending=False): This sorts the resulting sums of transaction amounts in descending order, so the customers who have spent the most are listed first. By default, ascending=False sorts in descending order (highest to lowest).
# .head(10):
# 
# .head(10): After sorting, this function selects the first 10 rows of the result. These will be the top 10 customers who have spent the most, according to the "Amount" column."""

# #which transaction have the highest transaction 

# In[77]:


df.groupby("TransactionID")["Amount"].sum().sort_values(ascending=False).head(30)


# #which city  have the highest transaction 

# In[78]:


df.groupby("Location")["Amount"].sum().sort_values(ascending=False).head()


# In[79]:


df["Location"].value_counts().head(5)


# In[80]:


df["Amount"].value_counts().head(5)


# In[81]:


df["CustomerID"].value_counts().head(5)


# """The code df["Location"].value_counts().head(5) is used to find the top 5 most frequent locations in the "Location" column of your DataFrame df. Let's break it down:
# 
# Line-by-line explanation:
# df["Location"]:
# 
# df["Location"]: This accesses the "Location" column in your DataFrame df. The "Location" column likely contains information about where the transaction took place, or the location of customers, etc.
# value_counts():
# 
# value_counts(): This function counts the number of occurrences of each unique value in the "Location" column. It returns a Series where the index is the unique locations, and the values are the count of how many times each location appears in the column. This allows you to see which locations are the most common.
# head(5):
# 
# head(5): After counting the occurrences of each location, this function selects the first 5 rows. Since the value_counts() function sorts the results by frequency (from highest to lowest), the first 5 will be the top 5 most frequent locations."""

# ### Fraud  Detection Analysis
# #how many fraudulent transactions are  there ?
# 

# In[82]:


df["isFraud"].value_counts()

Answer:total no of the fraudulents are 5065

# In[ ]:





# Q.what is the total fraud amount

# In[83]:


df[df["isFraud"] == 1]["Amount"].sum()


# In[84]:


df[df["isFraud"] == 1]["Amount"].sum()


# In[85]:


"""The code df[df["isFraud"] == 1]["Amount"].sum() is used to calculate the total transaction amount for fraudulent transactions in the "isFraud" column of your DataFrame df. Let's break down each part of this code:

Line-by-line explanation:
df[df["isFraud"] == 1]: Single equals (=): This is used for assignment. It assigns a value to a variable, like x = 5.

Double equals (==): This is used for comparison. It checks if two values are equal. In this case, it's checking whether each value in the isFraud column is equal to 1.

df["isFraud"] == 1: This creates a boolean condition that checks whether the value in the "isFraud" column is 1 (which we assume represents fraudulent transactions).
df[...]: This filters the DataFrame df to include only the rows where the condition df["isFraud"] == 1 is True. In other words, it selects only the rows where the transaction is fraudulent."""


# 
# df[df["isFraud"] == 1]["Location"].value_counts().head(10)

# #what is the fraud rate per transaction type?

# In[86]:


df.groupby("TransactionType")["isFraud"].mean()*100


# #what is the fraud rate per location?

# In[87]:


df.groupby("Location")["isFraud"].mean()*100


# In[88]:


# we use stns library here for the statistical representation and plt for the virtualize the data


# Q.Do fraudulent transaction have higher amounts?
# 

# In[89]:


sns.boxplot(data=df, x="isFraud",y ="Amount")
df.info()


# #Location wise amount of data

# In[90]:


sns.boxplot(data=df, x="Amount",y ="Location")
df.info()


# In[91]:


sns.boxplot(data=df , x="Amount",y="Location")
df.info()


# """The code sns.boxplot(data=df, x="isFraud", y="Amount") creates a boxplot using Seaborn to visualize the distribution of transaction amounts for fraudulent and non-fraudulent transactions. Let's break it down:
# 
# Line-by-line explanation:
# sns.boxplot(...):
# 
# sns.boxplot(...): This is a function from the Seaborn library that creates a boxplot, which is a useful way to display the distribution of a numerical variable and identify outliers. A boxplot shows the median, quartiles, and possible outliers of the data, making it easy to compare distributions between categories.
# data=df:
# 
# data=df: This specifies the DataFrame df that contains the data. The data argument tells Seaborn where to look for the data.
# x="isFraud":
# 
# x="isFraud": This sets the variable on the x-axis to "isFraud." The "isFraud" column typically contains binary values, such as 1 for fraudulent transactions and 0 for non-fraudulent transactions. The boxplot will create two categories on the x-axis: one for fraudulent transactions and one for non-fraudulent transactions.
# y="Amount":
# 
# y="Amount": This sets the variable on the y-axis to "Amount," which represents the transaction amount. This shows how the transaction amounts are distributed for each category of fraud (fraudulent and non-fraudulent)."""

# #what are the peak transaction hours
# 

# In[92]:


df["Hour"] = df["TransactionDate"].dt.hour
sns.countplot(x="Hour" , data=df)


# In[93]:


df.info()


# """1. df["Hour"] = df["TransactionDate"].dt.hour:
# df["TransactionDate"]: This refers to the "TransactionDate" column in your DataFrame df. The "TransactionDate" column likely contains timestamps of when each transaction occurred.
# .dt.hour: The .dt accessor is used to extract specific components from datetime-like data. By applying .hour, you extract the hour (from 0 to 23) from each timestamp in the "TransactionDate" column. This gives you the hour of the day when the transaction occurred.
# df["Hour"] = ...: This stores the extracted hour values in a new column called "Hour" in the DataFrame. Now, for each transaction, you have a column representing the hour at which the transaction took place.
# 2. sns.countplot(x="Hour", data=df):
# sns.countplot(...): This is a Seaborn function that creates a bar plot showing the frequency (count) of each unique value in the specified column. It is often used to visualize categorical data and how often each category appears.
# x="Hour": This specifies that the "Hour" column should be plotted on the x-axis. The "Hour" column contains the hours of the day when transactions occurred, so the plot will show the distribution of transactions by hour.
# data=df: This specifies the DataFrame df to use for the plot, which contains the "Hour" column."""

# Ans:peak time of transaction is 17 hrs means 5 PM

# In[94]:


df["Year"] = df["TransactionDate"].dt.year
sns.countplot(x="Year" , data=df)


# #### what is the fraud trend over time as per month

# In[95]:


df.groupby(df["TransactionDate"].dt.month)["isFraud"].sum().plot()


# In[96]:


Answer:Fraud spikes in september(9th month).


# """1. df["TransactionDate"].dt.month:
# df["TransactionDate"]: This refers to the "TransactionDate" column in your DataFrame, which likely contains datetime values.
# .dt.month: This extracts the month from the "TransactionDate" column. It returns an integer representing the month of each transaction, ranging from 1 (January) to 12 (December).
# 2. df.groupby(df["TransactionDate"].dt.month):
# groupby(df["TransactionDate"].dt.month): This groups the data by the month of the transaction. All transactions that happened in the same month will be grouped together. So, you’ll have 12 groups (one for each month), each representing the transactions for that specific month.
# 3. ["isFraud"]:
# ["isFraud"]: After grouping the data by month, we focus on the "isFraud" column, which likely contains 1 for fraudulent transactions and 0 for non-fraudulent ones.
# 4. .sum():
# .sum(): This sums the values of the "isFraud" column for each month. Since 1 represents fraudulent transactions, the sum gives the total number of fraudulent transactions for each month. For example, if a month has 3 fraudulent transactions, the sum for that month will be 3.
# 5. .plot():
# .plot(): This generates a plot (typically a line plot) to visualize the data. In this case, it will create a line plot showing the total number of fraudulent transactions for each month. The x-axis represents the months (from 1 to 12), and the y-axis represents the number of fraudulent transactions for each month."""

# In[ ]:


#what is the distribution of transaction by day of the week


# In[97]:


df["Dayofweek"] = df["TransactionDate"].dt.day_name()
sns.countplot(data=df, x="Dayofweek", order=["Monday","Tuesday","wednesday","Thursday","Friday","saturday","Sunday"])
plt.xticks(rotation=45)
plt.title("Transaction by day of the week")
plt.show()


# df["DayOfWeek"] = df["TransactionDate"].dt.day_name()
# """1. df["DayOfWeek"] = df["TransactionDate"].dt.day_name():
# df["TransactionDate"]: This refers to the "TransactionDate" column in your DataFrame df, which contains the datetime values for each transaction.
# .dt.day_name(): The .dt accessor allows you to extract specific components from the datetime values. .day_name() extracts the name of the day of the week (e.g., "Monday", "Tuesday", etc.) from each timestamp.
# df["DayOfWeek"] = ...: This stores the extracted day names in a new column called "DayOfWeek" in the DataFrame. Now, each transaction will have an associated "DayOfWeek" indicating the weekday on which it occurred."""
# sns.countplot(data=df, x="DayOfWeek", order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
# """2. sns.countplot(data=df, x="DayOfWeek", order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]):
# sns.countplot(...): This is a Seaborn function that creates a count plot (a type of bar plot) showing the frequency of each unique value in the "DayOfWeek" column.
# data=df: Specifies that we are using the df DataFrame for this plot.
# x="DayOfWeek": Specifies that the x-axis of the plot will represent the "DayOfWeek" column, so the bars will correspond to the days of the week.
# order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]: This ensures that the days of the week appear in the correct order, from Monday to Sunday, on the x-axis. Without specifying this order, the days might appear in an arbitrary order."""
# plt.xticks(rotation=45)
# """3. plt.xticks(rotation=45):
# plt.xticks(rotation=45): This rotates the x-axis labels (the day names) by 45 degrees for better readability. If the labels are too long and overlap, rotating them helps to make the plot clearer."""
# plt.title("Transactions by Day of the Week")
# """4. plt.title("Transactions by Day of the Week"):
# plt.title("Transactions by Day of the Week"): This sets the title of the plot to "Transactions by Day of the Week," giving context to what the plot represents."""
# plt.show()
# """5. plt.show():
# plt.show(): This displays the plot. Without this line, the plot might not appear depending on the environment you're working in."""

# In[98]:


df["DayOfWeek"] = df["TransactionDate"].dt.day_name()
sns.countplot(data=df, x="DayOfWeek", order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
plt.xticks(rotation=45)
plt.show()


# In[ ]:


print(df.dtypes)



# In[ ]:


df.corr(numeric_only=True)


# In[ ]:


non_numeric = df.select_dtypes(exclude=['number'])
print(non_numeric.head())


# In[ ]:


df = df.apply(pd.to_numeric, errors='coerce')  # Converts non-numeric to NaN


#  Q.what is the average transaction amount by day of the week?

# In[ ]:


#the column which we are extracting from is with [] and the thing what we are  extracting will be in ()


# In[99]:


df.groupby("Dayofweek")["Amount"].mean().sort_values(ascending=False)


# """1. df.groupby("DayOfWeek"):
# groupby("DayOfWeek"): This groups the data in your DataFrame df by the "DayOfWeek" column. The "DayOfWeek" column should contain the names of the days of the week (e.g., "Monday", "Tuesday", etc.). This operation creates a separate group for each unique day of the week.
# 2. ["Amount"]:
# ["Amount"]: After grouping the data by "DayOfWeek," we select the "Amount" column, which represents the transaction amounts. This allows us to perform operations specifically on the transaction amounts for each day.
# 3. .mean():
# .mean(): This calculates the average (mean) of the "Amount" values within each group (i.e., for each day of the week). For example, it will calculate the average transaction amount for all the transactions that occurred on Monday, the average amount for all the transactions on Tuesday, and so on.
# 4. .sort_values(ascending=False):
# .sort_values(ascending=False): This sorts the resulting means (average transaction amounts) in descending order. So, the days with the highest average transaction amounts will appear at the top of the output, and the days with the lowest average transaction amounts will appear at the bottom."""

# In[ ]:




