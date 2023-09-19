import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load your data and preprocess it as needed
data = pd.read_csv('../excercises/Data/house-data.csv', index_col=0)
# ... (your data preprocessing code here)

# Define the trainingStation function
def trainingStation(X, Xname):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=0.15)
    myreg = LinearRegression()
    myreg.fit(X_train, y_train)
    a = myreg.coef_
    b = myreg.intercept_
    y_predicted = myreg.predict(X_test)
    
    # Create a unique plot for each call to trainingStation
    plt.figure()
    plt.title('Linear Regression')
    plt.scatter(X, y, color='green')
    plt.plot(X_train, a*X_train + b, color='blue')
    plt.plot(X_test, y_predicted, color='orange')
    plt.xlabel(Xname)
    st.pyplot(plt)  # Display the plot using Streamlit's pyplot function

# Define the feature columns and target variable
# ...

# Streamlit app
def main():
    st.title('Linear Regression Model')
    st.sidebar.header('Parameters')
    # ...
    
    # Call the trainingStation function for each feature you want to visualize
    trainingStation(Xsqft_living, 'sqftliving')
    trainingStation(Xbedrooms, 'bedrooms')
    trainingStation(Xbathrooms, 'bathrooms')
    trainingStation(Xview, 'view')
    trainingStation(Xzip, 'zipcode')
    trainingStation(Xgrade, 'grade')
    trainingStation(Xrenovated, 'yr_renovated')

    st.write("There's about 20000 rows, so for the general idea of how it looks after removing unwanted columns, the first 5 rows are shown")

    st.write("Underneath a heatmap from ")

if __name__ == '__main__':
    main()
