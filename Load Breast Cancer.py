import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import seaborn as sns

data = load_breast_cancer()
print('data - shape', data.data.shape)

X = data.data
y = data.target

X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 random_state=42, test_size= 0.2)

lg = LogisticRegression()
lg.fit(X_train,y_train)

y_pred = lg.predict(X_test)

accuracy = accuracy_score(y_pred, y_test)
print(f"Model accuracy: {accuracy:.2f}")

print("Classification Report")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=data, yticklabels=data)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM Confusion Matrix - Iris")
plt.show()

input('wait...')
