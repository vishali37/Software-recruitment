import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('recruitment1.csv')

# Preprocess the data
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=False)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the training and test sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Display the results
st.title('Software Recruitment prediction Model')
input_features = st.text_input("Enter the Year")

# Check if input is provided
if input_features:
    # Convert the input to a NumPy array
    input_array = np.fromstring(input_features, sep=' ').reshape(-1,1)
    # Standardize the input
    input_array = scaler.transform(input_array)
    
    # Make the prediction
    prediction = model.predict(input_array)
    
    # Print the prediction
    st.write("The predicted value is:", prediction[0])
st.subheader('Training Results')
for i in range(len(y_train)):
    st.write(f"{X_train[i]}, {y_train[i]}, {y_train_pred[i]}")

st.subheader('Test Results')
for i in range(len(y_test)):
    st.write(f"{X_test[i]}, {y_test[i]}, {y_test_pred[i]}")

# Plot the results
plt.figure(figsize=(10, 5))
plt.scatter(X_train[:,0], y_train, color='green')
plt.scatter(X_train[:,0], y_train_pred, color='red')
plt.title('Year vs Recruitment (Training Set)')
plt.xlabel('year')
plt.ylabel('recruitment')
st.pyplot(plt)

plt.figure(figsize=(10,5))
plt.scatter(X_test[:,0], y_test, color='blue')
plt.scatter(X_test[:,0], y_test_pred, color='red')
plt.title('Year vs Recruitment (Test Set)')
plt.xlabel('year')
plt.ylabel('Recruitment')
st.pyplot(plt)