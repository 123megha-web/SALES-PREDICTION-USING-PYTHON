import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import streamlit as st

# Load the dataset
data = pd.read_csv('advertising.csv')  # Update with your dataset path

# Split features and target variable
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Streamlit app
st.title('Sales Prediction App')

# Input fields for user data
tv = st.number_input('TV Advertising Budget (in $)', min_value=0, step=1)
radio = st.number_input('Radio Advertising Budget (in $)', min_value=0, step=1)
newspaper = st.number_input('Newspaper Advertising Budget (in $)', min_value=0, step=1)

# Predict sales based on user input
if st.button('Predict Sales'):
    input_data = pd.DataFrame([[tv, radio, newspaper]], columns=['TV', 'Radio', 'Newspaper'])
    prediction = model.predict(input_data)
    st.write(f"Predicted Sales: ${prediction[0]:,.2f}")
    st.write(f"Mean Squared Error of the model: {mse:.2f}")

