import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier , plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()

print("load_iris.shape", iris.data.shape)
X = iris.data
y = iris.target
# Step 3: Split dataset into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Create and train the Decision Tree model
clf = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# Step 5: Predictions on the test set
y_pred = clf.predict(X_test)

# Step 6: Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

plt.figure(figsize=(12,8))
plot_tree(clf, 
          filled=True, 
          feature_names=iris.feature_names, 
          class_names=iris.target_names, 
          rounded=True)
plt.show()

y_pred = clf.predict(X)
print("Predictions for first 5 samples:", y_pred[:5])

y_pred = clf.predict(X_test)   # use X_test, not full X

from sklearn.metrics import classification_report
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))