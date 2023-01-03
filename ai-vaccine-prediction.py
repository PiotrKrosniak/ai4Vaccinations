# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load the dataset containing past data on vaccine demand and supply
data = pd.read_csv('vaccine_data.csv')

# Select the relevant columns to be used as features for the model
X = data[['month', 'region', 'population', 'prev_demand']]

# Select the column to be predicted (vaccine demand in the current month)
y = data['demand']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model using a random forest regressor
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Use the trained model to make predictions on the test set
predictions = model.predict(X_test)

# Calculate the accuracy of the predictions
accuracy = model.score(X_test, y_test)
print('Accuracy:', accuracy)

# Use the trained model to make predictions on future vaccine demand
future_predictions = model.predict(X_future)
