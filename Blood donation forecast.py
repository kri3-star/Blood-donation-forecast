import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('blood_donations.csv')

# Define features and target variable
X = data[['age', 'gender', 'number_of_donations_last_year', 'days_since_last_donation']]
y = data['total_donations']

# Handle categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age', 'number_of_donations_last_year', 'days_since_last_donation']),
        ('cat', OneHotEncoder(), ['gender'])
    ])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline that first preprocesses the data and then trains the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Plotting the predicted vs actual values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Donations')
plt.ylabel('Predicted Donations')
plt.title('Actual vs Predicted Donations')
plt.show()

# Example new data
new_data = pd.DataFrame({
    'age': [30, 45],
    'gender': ['Male', 'Female'],
    'number_of_donations_last_year': [1, 2],
    'days_since_last_donation': [100, 200]
})

# Make predictions
predictions = model.predict(new_data)
print("Predicted future donations:", predictions)
