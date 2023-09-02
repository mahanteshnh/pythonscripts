import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load the Iris dataset from scikit-learn

from sklearn.datasets import load_iris
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Map the target classes to binary: setosa (class 0) vs. not setosa (class 1)
data['target'] = (data['target'] == 0).astype(int)

# Split the data into features (X) and the target variable (y)
X = data[['sepal length (cm)', 'sepal width (cm)']]  # Features
y = data['target'] # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)


# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model's accuracy

accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')

# You can use this trained model to make predictions for new data
