from sklearn import tree


# Sample data: features (weight in grams and color) and labels (0 for apple, 1 for orange)

features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1]

# Create a decision tree classifier

clf = tree.DecisionTreeClassifier()


# Fit the model to the data

clf = clf.fit(features, labels)

# Make predictions
# Lets predict the class of the fruit that weighs 160gms and is red(color=0)
prediction = clf.predict([[180, 1]])


# Map the prediction back to the fruit name

fruit_names = {0: 'Apple', 1: 'Orange'}
predicted_fruit = fruit_names[prediction[0]]

print("Predicted fruit: ", predicted_fruit)


