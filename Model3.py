# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load your dataset (replace 'data.csv' with your data file)
data = pd.read_csv('data.csv')


# Preprocess the data into training and testing sets

X = data.drop('target_column', axis=1) # Features
y = data['target_column'] # Target Variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose the Model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the Model

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# You can further tune hyperparameters and fine-tune your model as needed.
