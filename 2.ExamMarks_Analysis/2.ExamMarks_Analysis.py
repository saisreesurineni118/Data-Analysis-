#!/usr/bin/env python
# coding: utf-8

# Dataset-class.marks.csv

# **Introduction**
# Student performance evaluation is essential in academic institutions to assess knowledge retention and learning outcomes. The class marks dataset consists of scores obtained by students in different questions of an exam. This dataset helps in analyzing students' strengths and weaknesses, identifying challenging questions, and understanding overall performance trends. By visualizing this data, we can gain insights into score distributions, common patterns in student performance, and areas that may require additional instructional support.

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
a = pd.read_csv("class_marks.csv")
a.fillna(0, inplace=True)
a["Q1"] = a["Q1aM4"] + a["Q1bM6"]
a["Q2"] = a["Q2aM6"] + a["Q2bM4"]
a["Q3"] = a["Q3aM5"] + a["Q3bM5"]
a["Q4"] = a["Q4aM3"] + a["Q4bM7"]
a["Q5"] = a["Q5M10"]
a["Q6"] = a["Q6aM4"] + a["Q6bM6"]
columns_to_drop = [
    "Q1aM4", "Q1bM6", "Q2aM6", "Q2bM4", 
    "Q3aM5", "Q3bM5", "Q4aM3", "Q4bM7", 
    "Q5M10", "Q6aM4", "Q6bM6"
]
a = a.drop(columns=columns_to_drop)
a = a.astype(int)
def assign_grade(total):
    if total >= 45:
        return 'A+'
    elif total >= 40:
        return 'A'
    elif total >= 35:
        return 'B'
    elif total >= 30:
        return 'C'
    elif total >= 25:
        return 'D'
    else:
        return 'F'
a['Grade'] = a['Total'].apply(assign_grade)
a


# **Column Description**
# Total - The total marks obtained by a student.
# Q1aM4 - Marks scored in question 1a (maximum 4 marks).
# Q1bM6 - Marks scored in question 1b (maximum 6 marks).
# Q2aM6 - Marks scored in question 2a (maximum 6 marks).
# Q2bM4 - Marks scored in question 2b (maximum 4 marks).
# Q3aM5 - Marks scored in question 3a (maximum 5 marks).
# Q3bM5 - Marks scored in question 3b (maximum 5 marks).
# Q4aM3 - Marks scored in question 4a (maximum 3 marks).
# Q4bM7 - Marks scored in question 4b (maximum 7 marks).
# Q5M10 - Marks scored in question 5 (maximum 10 marks).
# Q6aM4 - Marks scored in question 6a (maximum 4 marks).
# Q6bM6 - Marks scored in question 6b (maximum 6 marks).

# In[3]:


a.shape


# In[4]:


a.info()


# Processesing a dataset class_marks.csv, computes aggregated scores (Q1–Q6, TOTAL) from sub-columns, drops the original sub-columns, and retains only the processed, meaningful columns.

# In[5]:


a=a.sort_values("Total",ascending=True)
a


# The dataset `a` is sorted in ascending order based on the "Total" column, rearranging the rows so that entries with the lowest total scores appear first.

# **Data visualizations**

# In[6]:


from scipy.stats import skew
filtered_data = a.loc[(a['Total'] >= 25) & (a['Total'] <= 35)]
skewness_values = filtered_data[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']].apply(skew)
plt.figure(figsize=(8, 5))
skewness_values.plot(kind='bar', color='cornflowerblue', edgecolor='black')
plt.axhline(0, color='red', linestyle='--', linewidth=1, label="No Skewness")

plt.title("Skewness of Marks Distribution for Each Question")
plt.xlabel("Questions")
plt.ylabel("Skewness Value")
plt.xticks(rotation=0)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Code calculates the skewness of marks distribution for each question (Q1 to Q6) within a filtered range of total marks (25-35) and visualizes it using a bar plot
# Skewness is a measure of the asymmetry of a distribution, where:
# Positive skew indicates that the right tail (higher marks) is longer or fatter.
# Negative skew indicates that the left tail (lower marks) is longer or fatter.
# Zero skew indicates that the distribution is symmetric.
# Q1, Q2, and Q5:
# These questions have negative skewness, meaning the marks are skewed towards higher values (most students performed well).
# Q4 and Q6:
# These questions have positive skewness, meaning the marks are skewed towards lower values (most students performed poorly).
# Q3:
# Has almost zero skewness, indicating a nearly symmetric marks distribution.

# In[19]:


filtered_data = a.loc[(a['Total'] >= 25) & (a['Total'] <= 35)]
question_averages = filtered_data[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']].mean()
above_average_percentage = (
    filtered_data[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']] > question_averages
).mean() * 100

# Plot the percentages
plt.figure(figsize=(8, 5))
above_average_percentage.plot(kind='bar', color='yellow', edgecolor='black')
plt.title("Percentage of Students Scoring Above Average by Question")
plt.xlabel("Questions")
plt.ylabel("Percentage (%)")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# code filters students data for total marks between 25-35, calculates the percentage of students scoring above average per question (Q1-Q6), and visualizes the results as a bar plot.

# In[25]:


filtered_data = a.loc[(a['Total'] >= 10) & (a['Total'] <= 20)]
average_marks = filtered_data[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']].mean()

plt.figure(figsize=(8, 5))
average_marks.plot(kind='pie', autopct='%1.1f%%', colors=['lightcoral', 'red', 'yellow', 'gold', 'violet', 'orange'])
plt.title("Average Marks Distribution for Students with Total 10-20")
plt.ylabel("")  # Hide the y-label
plt.show()


# In[ ]:





# Visuals represents the average marks for each question (Q1-Q6) from students with total marks between 25-35, visualized as a bar plot with labeled axes and gridlines for better clarity.

# In[9]:


filtered_data = a.loc[(a['Total'] >= 25) & (a['Total'] <= 35)]
average_marks = filtered_data[['Q1', 'Q2']].mode().iloc[0]

plt.figure(figsize=(8, 5))
average_marks.plot(kind='bar', color='lightcoral', grid=False, edgecolor='black')

plt.title("Mode of Marks for Each Question (Total Scores Between 25 and 35)")
plt.xlabel("Questions")
plt.ylabel("Mode of Marks")
plt.xticks(ticks=range(len(average_marks.index)), labels=average_marks.index, rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# The code calculates and visualizes the mode of marks for each question (Q1-Q6) from students with total marks between 25-35, using a bar plot to represent the most frequent scores.

# In[20]:


max_marks = filtered_data[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']].max()
full_marks_count = (filtered_data[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']] == max_marks).sum()
plt.figure(figsize=(8, 5))
full_marks_count.plot(kind='bar', color='lightblue', edgecolor='black')
plt.title("Count of Students Scoring Full Marks by Question")
plt.xlabel("Questions")
plt.ylabel("Count of Full Marks")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Code displays the count of students scoring full marks for each question (Q1-Q6) among those with total marks between 25-35, visualized as a bar plot with labeled axes and gridlines.

# In[11]:


filtered_data = a.loc[(a['Total'] >= 25) & (a['Total'] <= 35)]
median_contribution= filtered_data[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']].median()
plt.figure(figsize=(8, 5))
median_contribution.plot(kind='bar', color='mediumpurple', edgecolor='black')
plt.title("Median Percentage Contribution of Each Question to Total Marks")
plt.xlabel("Questions")
plt.ylabel("Median Contribution (%)")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Code Illustrates the median percentage contribution of each question (Q1-Q6) to total marks for students with total marks between 25-35, displayed as a bar plot with labeled axes and gridlines.

# In[12]:


filtered_data = a.loc[(a['Total'] >= 25) & (a['Total'] <= 35)]
contribution = filtered_data[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']].div(filtered_data['Total'], axis=0) * 100
range_contribution = contribution.max() - contribution.min()
plt.figure(figsize=(8, 5))
range_contribution.plot(kind='bar', color='darkorange', edgecolor='black')
plt.title("Range of Contribution for Each Question to Total Marks")
plt.xlabel("Questions")
plt.ylabel("Range of Contribution (%)")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Code shows the range of contribution (maximum - minimum) of each question (Q1-Q6) to total marks for students with total marks between 25-35, visualized as a bar plot with gridlines.

# In[13]:


b=a.loc[(a['Total'] >= 10) & (a['Total'] <= 25)]
b.plot.scatter(x='Q1',y='Total',color='red',s=40)
plt.title("10-25 Marks ANALYSIS-Scatter plot")
plt.show()


# Scatter plot visualizes the relationship between Q1 marks and total marks for students scoring between 10-25. Red points represent individual data points, highlighting the distribution and correlation.

# In[28]:


b=a.loc[(a['Total'] >= 10) & (a['Total'] <= 20)]
b.plot.scatter(x='Q5',y='Total',color='lightblue',s=40)
plt.title("10-20 Marks ANALYSIS-Scatter plot")
plt.show()


# In[14]:


c=a.loc[(a['Total'] >= 25) & (a['Total'] <= 30)]
c.boxplot(by='Q3', column =['Total'], grid = False,color='orange',figsize=[5,6])
plt.title("25-30 Marks")
plt.ylabel("Total")
plt.show()


# Boxplot displays the distribution of total marks for students scoring between 25-30, grouped by Q3 marks. Orange color highlights the range, with labels indicating the total marks for each group.

# In[17]:


c=a.loc[(a['Total'] >= 20) & (a['Total'] <= 30)]
c.boxplot(by='Q1', column =['Total'], grid = False,color='orange',figsize=[5,6])
plt.title("20-30 Marks")
plt.ylabel("Total")
plt.show()


# In[15]:


a.plot.line(x='Q2',y='Total',color='red')
plt.title("Line Graph of Q2")
plt.ylabel("Total")
plt.show()


# In[ ]:





# This line graph shows the relationship between Q2 marks and total marks, with Q2 on the x-axis and total marks on the y-axis, using a red line to connect the data points.

# In[18]:


a.plot.line(x='Q5',y='Total',color='red')
plt.title("Line Graph of Q5")
plt.ylabel("Total")
plt.show()


# **Conclusion**
# The dataset analysis reveals variations in student performance across different questions. The total marks distribution suggests that a majority of students score in a mid-range, indicating an average level of understanding. Certain questions, such as Q4aM3 and Q6bM6, have a high percentage of missing scores, suggesting that students either skipped them or found them difficult. Additionally, long-answer questions like Q5M10 show more variation in scores, indicating differences in comprehension and expression among students.  
# From the visualization trends, we can infer that some questions may require better explanation or revision sessions, while others might need adjustments in grading or question design. This analysis provides a foundation for improving student performance assessment and refining the exam structure for future evaluations.
