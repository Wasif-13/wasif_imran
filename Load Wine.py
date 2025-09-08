import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

wine_data = load_wine()

print("wine_data - shape()", wine_data.data.shape)

wine_data = load_wine()
X,y = wine_data.data , wine_data.target
target_names = [str(i) for i in range(3)] 
# Split dataset into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the Decision Tree model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predictions on the test set
y_pred = log_reg.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")


y_pred = log_reg.predict(X)
print("Predictions samples:", y_pred)

y_pred = log_reg.predict(X_test)   # use X_test, not full X

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion matrix heatmap
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM Confusion Matrix - Iris")
plt.show()

input('wait...')