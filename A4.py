#!/usr/bin/env python
# coding: utf-8

# In[1]:



# In[34]:


# import
import streamlit as st
import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
#import graphviz
import seaborn as sns
import matplotlib.pyplot as plt



from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score




# In[3]:


df= pd.read_csv('../excercises/Data/diabetes.csv')




# In[5]:
##Frontend

#idea of general looks of data
st.write("This is how the data looks like in a dataframe. Note that it's not all of the data, it's just to give a general idea of what's going on")
st.write(df.head())
st.write("\n")  # Newline character to create a space
st.write("\n")  # Newline character to create a space

# In[6]:

st.write("We can also get a general overview of the data which is shown below")
st.write(df.describe())

st.write("The only odd thing about the data is a few 0-values in the data. We are going to try to standardize this.")

##Code
DPF='DiabetesPedigreeFunction'
#Making a median of some of the parameters, as there's some 0-values (not null values!) and replacing them in the dataframe
median_blood_pressure = df[df['BloodPressure'] != 0]['BloodPressure'].median()

df['BloodPressure'] = df['BloodPressure'].replace(0, median_blood_pressure)

median_BMI = df[df['BMI'] != 0]['BMI'].median()

df['BMI'] = df['BMI'].replace(0, median_BMI)

median_insulin = df[df['Insulin'] != 0]['Insulin'].median()

df['Insulin'] = df['Insulin'].replace(0, median_insulin)

median_glucose = df[df['Glucose'] != 0]['Glucose'].median()

df['Glucose'] = df['Glucose'].replace(0, median_glucose)

median_skin_thickness = df[df['SkinThickness'] != 0]['SkinThickness'].median()

df['SkinThickness'] = df['SkinThickness'].replace(0, median_skin_thickness)

feature_cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
X = df[feature_cols]
y = df['Outcome']

X_standardized=(X-X.mean())/X.std()

##Frontend
st.write("\n")  # Newline character to create a space
st.write("\n")  # Newline character to create a space
st.write("Now everything has been standardized")
st.write(X_standardized)
##Code

X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.2, random_state=123)

df_without_outcome = df.drop(columns=['Outcome'])

# Set a larger figure size
plt.figure(figsize=(16, 4))  # Adjust the width (16) to your preference

df_without_outcome.plot(kind='box', subplots=True, layout=(1, 8), sharex=False, sharey=False)
firstFig= plt.show()
##Frontend
st.write("A few boxplots have been made. Due to an unknown error or missing skillset, it's hard to read the titles")
st.write("The order is the following: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age")

st.pyplot(firstFig)

st.write("Aaand a few more diagrams to give a more vizualized picture of the data")
##Code
df.hist()
Fig2= plt.show()
##Frontend
st.pyplot(Fig2)
##Code
test_set_size = 0.2


seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_standardized, y, test_size=test_set_size, random_state=seed)

model = GaussianNB()
model.fit(X_train, Y_train)

##Frontend

st.title("BAYES")
st.write("after dividing into training and testing, a bayes model score has been found which is seen below")
st.write(model.score(X_test, Y_test))

##Code
prediction = model.predict(X_test)

scoring = 'accuracy'


cmat = confusion_matrix(Y_test, prediction)

sns.set()
sns.heatmap(cmat.T, square=False, annot=True, cbar=False)
plt.xlabel('actual')
plt.ylabel('predicted');
##For some reason not printing all of the numbers in the heatmap?

fig3=plt.show()
##Frontend
st.write("We are aware of the missing values 2 of the boxes.")
st.write("The bottom left box should have the number '19' assigned to it, and the bottom right should have '37' assigned to it")
st.write("as seen below the true positive and true negative is much bigger compared to the false negative and false positive.")
st.pyplot(fig3)

st.write("\n")  # Newline character to create a space
st.write("\n")  # Newline character to create a space
st.title("DECISION TREE")
##Code

params = {'max_depth': 5}
classifier = DecisionTreeClassifier(**params)

 
classifier.fit(X_train, y_train)

#dot_data = tree.export_graphviz(classifier, out_file=None, 
#                         feature_names=df.columns[:8], class_names = True,        
#                         filled=True, rounded=True, proportion = False,
#                         special_characters=True)  
#graph = graphviz.Source(dot_data) 
#graph.render("tmp") 
#graph

##Frontend
st.write("graphwiz import aint working :(")

##Code

dt_classifier = DecisionTreeClassifier(random_state=123)
dt_classifier.fit(X_train, y_train)
# Predict on the test data
dt_predictions = dt_classifier.predict(X_test)

# Calculate the accuracy of the Decision Tree model
dt_accuracy = accuracy_score(y_test, dt_predictions)
##Frontend

st.write("\n")  # Newline character to create a space
st.write("\n")  # Newline character to create a space
st.write("Similar to the Bayes model, we also got an accuracy score from the decision tree")  # Newline character to create a space

st.write(dt_accuracy)
st.write("which as seen is relatively lower than the bayes score")
##Code
y_testp = classifier.predict(X_test)
confusion_mat = confusion_matrix(y_test,y_testp)

confusion = pd.crosstab(y_test,y_testp)

plt.imshow(confusion_mat, interpolation='nearest')
plt.title('Confusion matrix')
plt.colorbar()
ticks = np.arange(2)
plt.xticks(ticks, ticks)
plt.yticks(ticks, ticks)
plt.ylabel('True labels')
plt.xlabel('Predicted labels')

fig4=plt.show()
##Frontend

st.write("\n")  # Newline character to create a space
st.write("\n")  # Newline character to create a space
st.write("Finally we got a heatmap for the confusion matrix")  # Newline character to create a space
st.pyplot(fig4)
sns.heatmap(confusion_mat, square=False, annot=True, cbar=False)
plt.xlabel('actual')
plt.ylabel('predicted');
fig5=plt.show()
##Frontend
st.write("We are aware of the missing values 2 of the boxes.")
st.write("The bottom left box should have the number '53' assigned to it, and the bottom right should have '5' assigned to it")
st.pyplot(fig5)


