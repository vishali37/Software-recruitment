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

# Assuming the first column is 'Year' and the last column is 'Recruitment'
year_column = df.columns[0]  # The first column represents the year
recruitment_column = df.columns[-1]  # The last column represents recruitment

# Preprocess the data
X = df.iloc[:, :-1].values  # Features (all columns except the last one, which is recruitment)
y = df.iloc[:, -1].values   # Target variable (the last column, recruitment)

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the training and test sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Streamlit app
st.title('Software Recruitment Prediction Model')

# Input fields for user input, assuming one feature (Year) or more
input_features = st.text_input("Enter the feature values (e.g., 'Year' or 'Feature1 Feature2'): ")

# Check if input is provided
if input_features:
    # Convert the input to a NumPy array (handling multiple features if necessary)
    input_values = np.fromstring(input_features, sep=' ')
    
    # Reshape the input array for prediction (match number of features)
    input_array = input_values.reshape(1, -1)  # Reshape to 2D array
    
    # Standardize the input
    input_array = scaler.transform(input_array)
    
    # Make the prediction
    prediction = model.predict(input_array)
    
    # Display the prediction
    st.write("The predicted recruitment value is:", prediction[0])

# Check if the input year is in the dataset
def check_year_in_dataset(year_input):
    # Check if the year is in the dataset (in the first column)
    if year_input<1969:
        return False
    return True

# Function to check for recession (3 consecutive years of decreasing recruitment)
def check_recession(predictions):
    recession_points = []  # List to store indices where recession is detected
    for i in range(len(predictions) - 2):
        if predictions[i] > predictions[i+1] > predictions[i+2]:
            recession_points.append(i+1)  # Mark the middle of the decreasing trend
    return recession_points

# Check if there is a recession in the test set
recession_points = check_recession(y_test_pred)

# If recession is detected, display it
if recession_points:
    st.subheader('Recession Alert!')
    st.write("Recruitment is declining for three consecutive years. This indicates a potential recession.")
else:
    st.subheader('Recruitment Trend')
    st.write("No recession detected in the recruitment data.")

# Check if the year input is valid and exists in the dataset
year_input = st.number_input("Enter a Year for Prediction:", min_value=int(df[year_column].min()), max_value=int(df[year_column].max()) + 1, step=1)

# If the input year is not in the dataset, prompt the user to enter a future year
if not check_year_in_dataset(year_input):
    st.warning("Year not found in the dataset. Please enter a future year for prediction.")
else:
    # Prepare the year input as an array for prediction (it needs to be transformed to the correct shape)
    year_input_array = np.array([[year_input]])
    
    # Standardize the input
    year_input_array = scaler.transform(year_input_array)
    
    # Make the prediction for the entered year
    prediction = model.predict(year_input_array)
    
    # Display the prediction
    st.write(f"The predicted recruitment value for the year {year_input} is: {prediction[0]}")

    # Plot the results
    plt.figure(figsize=(10, 5))

    # Training Set Plot
    plt.scatter(df.iloc[:len(X_train), 0], y_train, color='green', label='Actual values (Training set)')
    plt.scatter(df.iloc[:len(X_train), 0], y_train_pred, color='red', label='Predicted values (Training set)')

    # Test Set Plot
    plt.scatter(df.iloc[:len(X_test), 0], y_test, color='blue', label='Actual values (Test set)')
    plt.scatter(df.iloc[:len(X_test), 0], y_test_pred, color='red', label='Predicted values (Test set)')

    # Highlight recession points in the test set
    if recession_points:
        plt.scatter(df.iloc[recession_points, 0], y_test[recession_points], color='orange', label='Recession Points', s=100, marker='X')

    # Add the entered year prediction to the plot
    plt.scatter(year_input, prediction, color='purple', s=150, label=f'Prediction for {year_input}', marker='*')

    # Set labels and title
    plt.title('Year vs Recruitment (Training & Test Set) with Year Prediction')
    plt.xlabel('Year')
    plt.ylabel('Recruitment')
    plt.legend()

    # Show the plot
    st.pyplot(plt)
